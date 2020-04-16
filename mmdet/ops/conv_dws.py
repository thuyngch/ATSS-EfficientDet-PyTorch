from torch import nn


class DepthwiseSeparableConv2d(nn.Module):
	def __init__(self, in_channels, out_channels, kernel_size, stride=1,
			padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros'):

		super(DepthwiseSeparableConv2d, self).__init__()

		self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size,
			stride=stride, padding=padding, dilation=dilation,
			groups=in_channels, bias=False, padding_mode=padding_mode)

		self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=bias)

		self.in_channels = in_channels
		self.out_channels = out_channels
		self.kernel_size = kernel_size
		self.stride = stride
		self.padding = padding
		self.dilation = dilation
		self.groups = groups
		self.bias = bias
		self.padding_mode = self.depthwise.padding_mode
		self.transposed = self.depthwise.transposed
		self.output_padding = self.depthwise.output_padding

	def forward(self, x):
		out = self.depthwise(x)
		out = self.pointwise(out)
		return out
