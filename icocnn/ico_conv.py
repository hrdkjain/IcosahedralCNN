import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class _IcoBase(nn.Module):
    def __init__(self, in_features, out_features, in_subdivisions, out_subdivisions, in_feature_depth, out_feature_depth, corner_mode):
        super(_IcoBase, self).__init__()

        assert corner_mode == 'zeros' or (corner_mode == 'average' and out_subdivisions > 0)

        self.in_features = in_features
        self.in_feature_depth = in_feature_depth
        self.in_channels = in_features * self.in_feature_depth
        self.out_features = out_features
        self.out_feature_depth = out_feature_depth
        self.out_channels = out_features * self.out_feature_depth
        self.stride = 1 if in_subdivisions == out_subdivisions else 2
        self.in_subdivisions = in_subdivisions
        self.out_subdivisions = out_subdivisions

        self.base_height_in = 2 ** self.in_subdivisions
        self.height_in = self.base_height_in * 5
        self.width_in = self.base_height_in * 2

        self.padded_base_height_in = self.base_height_in + 2
        self.padded_height_in = self.padded_base_height_in * 5
        self.padded_width_in = self.width_in + 2
        self.padded_width_half_in = self.padded_width_in // 2

        self.base_height_out = self.base_height_in // self.stride
        self.height_out = self.base_height_out * 5
        self.width_out = self.width_in // self.stride

        self.padded_base_height_out = self.base_height_out + 2
        self.padded_height_out = self.padded_base_height_out * 5
        self.padded_width_out = self.width_out + 2
        self.padded_width_half_out = self.padded_width_out // 2

        self.corner_mode = corner_mode

        self.pcorner_handle = None
        self.outcorner_handle = None

        # indexing for corner correction on output of layer (after convolution or upsampling)
        # corner handling in padding and output
        corner_target_y = (torch.arange(1, 6)[:, None] * self.base_height_out - 1).repeat(2, 1)
        corner_target_x = torch.tensor([0] * 5 + [self.base_height_out] * 5)[:, None]
        self.register_buffer('corner_target_y', corner_target_y)
        self.register_buffer('corner_target_x', corner_target_x)

        if self.corner_mode == 'zeros':
            self.outcorner_handle = self.set_outcorners_zero
        if self.corner_mode == 'average':
            # corner averaging
            # pixel top center
            y_tc = corner_target_y - 1
            x_tc = corner_target_x

            # pixel top right
            y_tr = corner_target_y - 1
            x_tr = corner_target_x + 1

            # pixel right
            y_r = corner_target_y
            x_r = corner_target_x + 1

            # pixel bottom left (from previous map)
            y_bl = (torch.arange(5)[:, None] * self.base_height_out).roll(-1, 0).repeat(2, 1)
            x_bl = torch.tensor([self.base_height_out - 1] * 5 + [self.width_out - 1] * 5)[:, None]

            # pixel bottom left (from previous map)
            y_l_0 = y_bl[:5]
            x_l_0 = x_bl[:5] + 1
            y_l_1 = corner_target_y[5:]
            x_l_1 = corner_target_x[5:] - 1
            y_l = torch.cat((y_l_0, y_l_1), 0)
            x_l = torch.cat((x_l_0, x_l_1), 0)

            corner_src_y = torch.cat((y_tc, y_tr, y_r, y_bl, y_l), 1)
            corner_src_x = torch.cat((x_tc, x_tr, x_r, x_bl, x_l), 1)
            self.register_buffer('corner_src_y', corner_src_y)
            self.register_buffer('corner_src_x', corner_src_x)

            self.outcorner_handle = self.set_outcorners_average


    def setup_g_padding(self):
        # channel shifting
        if self.in_feature_depth == 6:
            ch_shift_up = torch.arange(self.in_channels).view(-1, 6).roll(-1, 1).view(-1, 1)
            self.register_buffer('ch_shift_up', ch_shift_up)
            ch_shift_down = torch.arange(self.in_channels).view(-1, 6).roll(1, 1).view(-1, 1)
            self.register_buffer('ch_shift_down', ch_shift_down)

        # length of padding strips
        pad_len_long = self.base_height_in
        pad_len_short = pad_len_long - 1
        pad_len_variable = pad_len_long
        if self.corner_mode == 'zeros':
            pad_len_variable = pad_len_short

        # padding indices
        prev_map2 = [4, 0, 1, 2, 3]
        next_map2 = [1, 2, 3, 4, 0]

        padding_src_x = []
        padding_src_y = []
        padding_target_x = []
        padding_target_y = []

        padding_ch_up_src_x = []
        padding_ch_up_src_y = []
        padding_ch_up_target_x = []
        padding_ch_up_target_y = []

        padding_ch_down_src_x = []
        padding_ch_down_src_y = []
        padding_ch_down_target_x = []
        padding_ch_down_target_y = []

        for i in range(0, 5):
            # channel shift up padding
            # top left padding
            padding_ch_up_target_y += [i * self.padded_base_height_in] * pad_len_variable
            padding_ch_up_target_x += list(range(2, 2 + pad_len_variable))
            padding_ch_up_src_y += list(range(prev_map2[i] * self.base_height_in, prev_map2[i] * self.base_height_in + pad_len_variable))
            padding_ch_up_src_x += [0] * pad_len_variable

            # bottom right padding
            padding_ch_up_target_y += [(i + 1) * self.padded_base_height_in - 1] * pad_len_long
            padding_ch_up_target_x += list(range(self.padded_width_half_in, self.padded_width_half_in + pad_len_long))
            padding_ch_up_src_y += list(range(next_map2[i] * self.base_height_in, next_map2[i] * self.base_height_in + pad_len_long))
            padding_ch_up_src_x += [-1] * pad_len_long

            # channel shift down padding
            # right padding
            padding_ch_down_target_y += list(range(i * self.padded_base_height_in + 1, i * self.padded_base_height_in + 1 + pad_len_short))
            padding_ch_down_target_x += [-1] * pad_len_short
            padding_ch_down_src_y += [(prev_map2[i] + 1) * self.base_height_in - 1] * pad_len_short
            padding_ch_down_src_x += list(range(self.base_height_in + 1, self.base_height_in + 1 + pad_len_short))

            # left padding
            padding_ch_down_target_y += list(range(i * self.padded_base_height_in + 1, i * self.padded_base_height_in + 1 + pad_len_long))
            padding_ch_down_target_x += [0] * pad_len_long
            padding_ch_down_src_y += [next_map2[i] * self.base_height_in] * pad_len_long
            padding_ch_down_src_x += list(range(0, pad_len_long))

            # no channel shift padding
            # top right padding
            padding_target_y += [i * self.padded_base_height_in] * pad_len_variable
            padding_target_x += list(range(self.padded_width_half_in + 1, self.padded_width_half_in + 1 + pad_len_variable))
            padding_src_y += [(prev_map2[i] + 1) * self.base_height_in - 1] * pad_len_variable
            padding_src_x += list(range(1, 1 + pad_len_variable))

            # bottom left padding
            padding_target_y += [(i + 1) * self.padded_base_height_in - 1] * pad_len_long
            padding_target_x += list(range(1, 1+ pad_len_long))
            padding_src_y += [next_map2[i] * self.base_height_in] * pad_len_long
            padding_src_x += list(range(self.base_height_in, self.base_height_in + pad_len_long))

        if self.in_feature_depth == 6:
            self.register_buffer('padding_ch_up_target_y', torch.LongTensor(padding_ch_up_target_y))
            self.register_buffer('padding_ch_up_target_x', torch.LongTensor(padding_ch_up_target_x))
            self.register_buffer('padding_ch_up_src_y', torch.LongTensor(padding_ch_up_src_y))
            self.register_buffer('padding_ch_up_src_x', torch.LongTensor(padding_ch_up_src_x))

            self.register_buffer('padding_ch_down_target_y', torch.LongTensor(padding_ch_down_target_y))
            self.register_buffer('padding_ch_down_target_x', torch.LongTensor(padding_ch_down_target_x))
            self.register_buffer('padding_ch_down_src_y', torch.LongTensor(padding_ch_down_src_y))
            self.register_buffer('padding_ch_down_src_x', torch.LongTensor(padding_ch_down_src_x))

            self.register_buffer('padding_target_y', torch.LongTensor(padding_target_y))
            self.register_buffer('padding_target_x', torch.LongTensor(padding_target_x))
            self.register_buffer('padding_src_y', torch.LongTensor(padding_src_y))
            self.register_buffer('padding_src_x', torch.LongTensor(padding_src_x))
        else:
            padding_target_y += padding_ch_up_target_y + padding_ch_down_target_y
            padding_target_x += padding_ch_up_target_x + padding_ch_down_target_x
            padding_src_y += padding_ch_up_src_y + padding_ch_down_src_y
            padding_src_x += padding_ch_up_src_x + padding_ch_down_src_x

            self.register_buffer('padding_target_y', torch.LongTensor(padding_target_y))
            self.register_buffer('padding_target_x', torch.LongTensor(padding_target_x))
            self.register_buffer('padding_src_y', torch.LongTensor(padding_src_y))
            self.register_buffer('padding_src_x', torch.LongTensor(padding_src_x))

        row_target = (torch.arange(1, self.padded_base_height_in - 1) +
                      torch.arange(0, self.padded_base_height_in * 5, self.padded_base_height_in)[:, None]).flatten()
        self.register_buffer('row_target', row_target)


        # corner handling in padding
        if self.corner_mode == 'zeros':
            # corner indices
            pcorner_x = torch.tensor([1, self.padded_width_half_in])
            self.register_buffer('pcorner_x', pcorner_x)
            pcorner_y = torch.arange(1, 6)[:, None] * self.padded_base_height_in - 2
            self.register_buffer('pcorner_y', pcorner_y)

            self.pcorner_handle = self.set_pcorners_zero
        else:
            # corner averaging for very top and bottom corner of icosahedron
            top_corner_target_y = torch.arange(5) * self.padded_base_height_in
            top_corner_target_x = torch.tensor([1])
            top_corner_src_y = torch.arange(5) * self.base_height_in
            top_corner_src_x = torch.tensor([0])

            bottom_corner_target_y = torch.arange(1, 6) * self.padded_base_height_in - 2
            bottom_corner_target_x = torch.tensor([-1])
            bottom_corner_src_y = torch.arange(1, 6) * self.base_height_in - 1
            bottom_corner_src_x = torch.tensor([-1])

            ext_corners_target_y = torch.stack((top_corner_target_y, bottom_corner_target_y))
            ext_corners_target_x = torch.stack((top_corner_target_x, bottom_corner_target_x))

            ext_corners_src_y = torch.stack((top_corner_src_y, bottom_corner_src_y))
            ext_corners_src_x = torch.stack((top_corner_src_x, bottom_corner_src_x))

            self.register_buffer('ext_corners_target_y', ext_corners_target_y)
            self.register_buffer('ext_corners_target_x', ext_corners_target_x)
            self.register_buffer('ext_corners_src_y', ext_corners_src_y)
            self.register_buffer('ext_corners_src_x', ext_corners_src_x)

            self.pcorner_handle = self.set_pcorners_average

    def set_pcorners_zero(self, outputs, inputs):
        outputs[:, :, self.pcorner_y, self.pcorner_x] = 0

    def set_pcorners_average(self, outputs, inputs):
        outputs[:, :, self.ext_corners_target_y, self.ext_corners_target_x] = torch.mean(
            inputs[:, :, self.ext_corners_src_y, self.ext_corners_src_x], -1, True)

    def set_outcorners_zero(self, outputs):
        outputs[:, :, self.corner_target_y, self.corner_target_x] = 0

    def set_outcorners_average(self, outputs):
        outputs[:, :, self.corner_target_y, self.corner_target_x] = torch.mean(
            outputs[:, :, self.corner_src_y, self.corner_src_x], -1, True)

    def g_pad(self, inputs):
        outputs = inputs.new_zeros(inputs.size(0), self.in_channels, self.padded_height_in, self.padded_width_in)

        # do the padding
        if self.in_feature_depth == 6:
            outputs[:, :, self.padding_target_y, self.padding_target_x] = inputs[:, :, self.padding_src_y, self.padding_src_x]
            outputs[:, :, self.padding_ch_up_target_y, self.padding_ch_up_target_x] = inputs[:, self.ch_shift_up, self.padding_ch_up_src_y, self.padding_ch_up_src_x]
            outputs[:, :, self.padding_ch_down_target_y, self.padding_ch_down_target_x] = inputs[:, self.ch_shift_down, self.padding_ch_down_src_y, self.padding_ch_down_src_x]
        else:
            outputs[:, :, self.padding_target_y, self.padding_target_x] = inputs[:, :, self.padding_src_y, self.padding_src_x]

        # copy maps from input
        outputs.narrow(3, 1, self.padded_width_in-2).index_copy_(2, self.row_target, inputs)

        # deal with corners according to selected mode
        self.pcorner_handle(outputs, inputs)

        return outputs

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def extra_repr(self):
        s = ('in_features={in_features}, '
             'in_feature_depth={in_feature_depth}, '
             'in_channels={in_channels}, '
             'out_features={out_features}, '
             'out_feature_depth={out_feature_depth}, '
             'out_channels={out_channels}, '
             'subdivisions={in_subdivisions}, '
             'stride={stride}')
        if self.bias is None:
            s += ', bias=False'
        return s.format(**self.__dict__)


class _IcoConv(_IcoBase):
    def __init__(self, in_features, out_features, bias, in_subdivisions, out_subdivisions, in_feature_depth, out_feature_depth, kernel_size, corner_mode):
        super(_IcoConv, self).__init__(
            in_features, out_features,
            in_subdivisions, out_subdivisions, in_feature_depth, out_feature_depth, corner_mode)

        # sanity checks
        assert kernel_size == 1 or kernel_size == 3

        # convolution specific params
        self.kernel_size = kernel_size
        weights_per_filter = 7 if kernel_size == 3 else 1

        # padding for convolution based on stride
        self.padding = (1, 0) if self.stride == 2 else (0, 0)

        # create filter weights
        self.weight = nn.Parameter(torch.empty(self.out_features, self.in_channels, weights_per_filter))

        # prepare filter expansion
        self.setup_filter_expansion(weights_per_filter)

        # bias
        if bias:
            self.bias = nn.Parameter(torch.Tensor(self.out_features, 1))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

        if self.kernel_size == 3:
            # prepare g_padding
            self.setup_g_padding()

            # row_indices that need copying after strided convolution (source and target)
            conv_base_height_out = self.padded_base_height_in // self.stride
            inner_padding_after_conv = 2 // self.stride
            row_offset_after_conv = 1 if self.stride == 2 else 0
            strided_row_src = (torch.arange(conv_base_height_out - inner_padding_after_conv) +
                                         torch.arange(0, conv_base_height_out * 5, conv_base_height_out)[:, None]).flatten() + row_offset_after_conv
            self.register_buffer('strided_row_src', strided_row_src)

    def setup_filter_expansion(self, weights_per_filter):

        # indices for filter creation: rows are clockwise rotations
        if weights_per_filter == 7:
            # for hexagonal kernel
            weight_indices = torch.tensor([[0, 1, 2, 3, 4, 5, 6],
                                           [2, 0, 5, 3, 1, 6, 4],
                                           [5, 2, 6, 3, 0, 4, 1],
                                           [6, 5, 4, 3, 2, 1, 0],
                                           [4, 6, 1, 3, 5, 0, 2],
                                           [1, 4, 0, 3, 6, 2, 5]]).type(torch.LongTensor)
        else:
            # for 1x1 kernel
            weight_indices = torch.tensor([[0], [0], [0], [0], [0], [0]]).type(torch.LongTensor)

        if self.in_feature_depth == 6:
            channel_indices = torch.zeros([self.in_feature_depth, self.in_channels], dtype=torch.long)
            base_channels = torch.arange(0, self.in_feature_depth)
            feature_starts = torch.arange(0, self.in_channels, self.in_feature_depth).reshape([-1, 1])
            for i in range(0, self.in_feature_depth):
                channel_indices[i] = (base_channels.roll(i) + feature_starts).reshape([-1])
        else:
            channel_indices = torch.arange(self.in_channels).expand(6, -1)

        k = torch.arange(self.in_channels * weights_per_filter).view(self.in_channels, weights_per_filter)
        if self.out_feature_depth == 6:
            l = torch.stack((
                k[channel_indices[0]].index_select(1, weight_indices[0]),
                k[channel_indices[1]].index_select(1, weight_indices[1]),
                k[channel_indices[2]].index_select(1, weight_indices[2]),
                k[channel_indices[3]].index_select(1, weight_indices[3]),
                k[channel_indices[4]].index_select(1, weight_indices[4]),
                k[channel_indices[5]].index_select(1, weight_indices[5]),
            ))
        else:
            l = k[channel_indices[0]].index_select(1, weight_indices[0]).unsqueeze(0)

        m = l
        for i in range(1, self.out_features):
            m = torch.cat((m, l + i * self.in_channels * weights_per_filter), 0)
        self.register_buffer('expanded_weight_indices', m)

    def compose_filters(self, filters, bias_expanded=None):
        # create suitable view of filters
        if self.kernel_size == 3:
            filters_flattened = filters.view([self.out_channels, self.in_channels, -1]).narrow(2, 1, 7)
        else:
            filters_flattened = filters.view([self.out_channels, self.in_channels, -1])
        filters_flattened.copy_(self.weight.take(self.expanded_weight_indices))

        if self.bias is not None:
            if self.out_feature_depth == 6:
                bias_expanded.copy_(self.bias.expand(-1, 6).flatten())
            else:
                bias_expanded.copy_(self.bias.flatten())

    def _forward(self, inputs):
        # compose filters & bias
        filters = inputs.new_zeros([self.out_channels, self.in_channels, self.kernel_size, self.kernel_size])
        bias_expanded = None
        if self.bias is not None:
            bias_expanded = inputs.new_empty(self.out_channels)
        self.compose_filters(filters, bias_expanded)

        # pad input
        if self.kernel_size == 3:
            inputs = self.g_pad(inputs)

        # convolution
        outputs = F.conv2d(inputs, filters, bias_expanded, self.stride, self.padding)

        # select the relevant rows (ignore padding rows)
        if self.kernel_size == 3:
            outputs = outputs.index_select(2, self.strided_row_src)
        else:
            if self.stride == 2:
                outputs = outputs.narrow(2, 1, self.height_out)

        # deal with corners according to chosen mode (set to zero or average)
        self.outcorner_handle(outputs)

        return outputs

    def forward(self, inputs):
        return self._forward(inputs)


class _IcoUpsample(_IcoBase):
    def __init__(self, in_features, out_features, in_subdivisions, out_subdivisions, in_feature_depth, out_feature_depth, corner_mode):
        super(_IcoUpsample, self).__init__(
            in_features, out_features,
            in_subdivisions, out_subdivisions, in_feature_depth, out_feature_depth, corner_mode)

        # general
        factor = 2

        # g_padding
        self.setup_g_padding()

        # upsampling parameters
        self.size = (self.padded_height_in * factor - 1, self.padded_width_in * factor - 1)
        self.scale_factor = None
        self.mode = 'bilinear'
        self.align_corners = True

        # output extraction
        # width
        self.col_start = 2
        self.col_length = self.width_in * factor

        # height
        # select rows to extract

        row_offset = 1  # ignore first row
        base_height_out = self.base_height_in * factor
        padded_base_height_out = self.padded_base_height_in * factor
        out_row_indices = (torch.arange(base_height_out) +
                       torch.arange(0, 5*padded_base_height_out, padded_base_height_out)[:,
                       None]).flatten() + row_offset
        self.register_buffer('out_row_indices', out_row_indices)

    def forward(self, inputs):

        # pad input
        inputs = self.g_pad(inputs)

        # upsample
        outputs = F.interpolate(inputs, self.size, self.scale_factor, self.mode, self.align_corners)

        # extract relevant pixels
        outputs = outputs.narrow(3, self.col_start, self.col_length).index_select(2, self.out_row_indices)

        # deal with corners according to chosen mode (set to zero or average)
        self.outcorner_handle(outputs)

        return outputs


class IcoBatchNorm(nn.BatchNorm3d):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True):
        super(IcoBatchNorm, self).__init__(num_features, eps, momentum, affine, track_running_stats)
        self.D = torch.tensor(6)

    def forward(self, input):
        N, C, H, W = input.size()
        D = self.D
        #assert torch.remainder(C, D) == 0
        G = C // D

        # reshape to 3D data
        input = input.view(N, G, D, H, W)

        # normalize
        output = super(IcoBatchNorm, self).forward(input)

        # revert reshaping
        output = output.view(N, C, H, W)

        return output


class IcoConvS2S(_IcoConv):
    def __init__(self, in_features, out_features, stride=1, bias=True, subdivisions=0, kernel_size=3, corner_mode='zeros'):
        super(IcoConvS2S, self).__init__(
            in_features, out_features, bias=bias,
            in_subdivisions=subdivisions,
            out_subdivisions=subdivisions if stride == 1 else subdivisions - 1,
            in_feature_depth=1, out_feature_depth=1, kernel_size=kernel_size, corner_mode=corner_mode)


class IcoConvS2R(_IcoConv):
    def __init__(self, in_features, out_features, stride=1, bias=True, subdivisions=0, kernel_size=3, corner_mode='zeros'):
        super(IcoConvS2R, self).__init__(
            in_features, out_features, bias=bias,
            in_subdivisions=subdivisions,
            out_subdivisions=subdivisions if stride == 1 else subdivisions - 1,
            in_feature_depth=1, out_feature_depth=6, kernel_size=kernel_size, corner_mode=corner_mode)


class IcoConvR2R(_IcoConv):
    def __init__(self, in_features, out_features, stride=1, bias=True, subdivisions=0, kernel_size=3, corner_mode='zeros'):
        super(IcoConvR2R, self).__init__(
            in_features, out_features, bias=bias,
            in_subdivisions=subdivisions,
            out_subdivisions=subdivisions if stride == 1 else subdivisions - 1,
            in_feature_depth=6, out_feature_depth=6, kernel_size=kernel_size, corner_mode=corner_mode)


class IcoUpsampleS2S(_IcoUpsample):
    def __init__(self, features, in_subdivisions=0, corner_mode='zeros'):
        super(IcoUpsampleS2S, self).__init__(
            features, features,
            in_subdivisions=in_subdivisions,
            out_subdivisions=in_subdivisions + 1,
            in_feature_depth=1, out_feature_depth=1, corner_mode=corner_mode)


class IcoUpsampleR2R(_IcoUpsample):
    def __init__(self, features, in_subdivisions=0, corner_mode='zeros'):
        super(IcoUpsampleR2R, self).__init__(
            features, features,
            in_subdivisions=in_subdivisions,
            out_subdivisions=in_subdivisions + 1,
            in_feature_depth=6, out_feature_depth=6, corner_mode=corner_mode)
