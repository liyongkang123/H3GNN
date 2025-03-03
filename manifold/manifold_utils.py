import torch

import sys
sys.path.append("..")
import config

args=config.parse()
device = torch.device('cuda:'+args.cuda if torch.cuda.is_available() else 'cpu')

def th_atanh(x, EPS):
    values = torch.min(x, torch.Tensor([1.0 - EPS]).to(x.device) )
    return 0.5 * (torch.log(1 + values + EPS) - torch.log(1 - values + EPS))
def th_norm(x, dim=1):
    """
	Args
		x: [batch size, dim]
	Output:
		[batch size, 1]
	"""
    return torch.norm(x, 2, dim, keepdim=True)


def th_dot(x, y, keepdim=True):
    return torch.sum(x * y, dim=1, keepdim=keepdim)


def clip_by_norm(x, clip_norm):
    return torch.renorm(x, 2, 0, clip_norm)

def clamp_max(x, max_value):
	t = torch.clamp(max_value - x.detach(), max=0)
	return x + t

def clamp_min(x, min_value):
	t = torch.clamp(min_value - x.detach(), min=0)
	return x + t

def one_hot_vec(length, pos):
	vec = [0] * length
	vec[pos] = 1
	return vec
def cosh(x, clamp=15):
    return x.clamp(-clamp, clamp).cosh()


def sinh(x, clamp=15):
    return x.clamp(-clamp, clamp).sinh()


def tanh(x, clamp=15):
    return x.clamp(-clamp, clamp).tanh()


def arcosh(x):
    return Arcosh.apply(x)


def arsinh(x):
    return Arsinh.apply(x)


def artanh(x):
    return Artanh.apply(x)


class Artanh(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        x = x.clamp(-1 + 1e-15, 1 - 1e-15)
        ctx.save_for_backward(x)
        z = x.double()
        return (torch.log_(1 + z).sub_(torch.log_(1 - z))).mul_(0.5).to(x.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        return grad_output / (1 - input ** 2)


class Arsinh(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        z = x.double()
        return (z + torch.sqrt_(1 + z.pow(2))).clamp_min_(1e-15).log_().to(x.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        return grad_output / (1 + input ** 2) ** 0.5


class Arcosh(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        x = x.clamp(min=1.0 + 1e-15)
        ctx.save_for_backward(x)
        z = x.double()
        return (z + torch.sqrt_(z.pow(2) - 1)).clamp_min_(1e-15).log_().to(x.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        return grad_output / (input ** 2 - 1) ** 0.5