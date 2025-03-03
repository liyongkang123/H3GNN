import torch
import torch.nn as nn, torch.nn.functional as F
from dgl.nn.pytorch import TypedLinear
import config
import math
from torch_geometric.nn import LayerNorm
from torch_scatter import scatter
from torch_geometric.utils import softmax
from torch.autograd import Function, Variable
from manifold.manifold_utils import *
'''
Implementation of two Poincaré models

The hyperbolic model has constant negative curvature.
'''

EPS = 1e-15
clip_value = 0.9899


class PoincareDistance(Function):
    @staticmethod
    def grad(x, v, sqnormx, sqnormv, sqdist, eps):
        alpha = (1 - sqnormx)
        beta = (1 - sqnormv)
        z = 1 + 2 * sqdist / (alpha * beta)
        a = ((sqnormv - 2 * torch.sum(x * v, dim=-1) + 1) / torch.pow(alpha, 2)) \
            .unsqueeze(-1).expand_as(x)
        a = a * x - v / alpha.unsqueeze(-1).expand_as(v)
        z = torch.sqrt(torch.pow(z, 2) - 1)
        z = torch.clamp(z * beta, min=eps).unsqueeze(-1)
        return 4 * a / z.expand_as(x)

    @staticmethod
    def forward(ctx, u, v, eps):
        squnorm = torch.clamp(torch.sum(u * u, dim=-1), 0, 1 - eps)
        sqvnorm = torch.clamp(torch.sum(v * v, dim=-1), 0, 1 - eps)
        sqdist = torch.sum(torch.pow(u - v, 2), dim=-1)
        ctx.eps = eps
        ctx.save_for_backward(u, v, squnorm, sqvnorm, sqdist)
        x = sqdist / ((1 - squnorm) * (1 - sqvnorm)) * 2 + 1
        # arcosh
        z = torch.sqrt(torch.pow(x, 2) - 1)
        return torch.log(x + z)

    @staticmethod
    def backward(ctx, g):
        u, v, squnorm, sqvnorm, sqdist = ctx.saved_tensors
        g = g.unsqueeze(-1)
        gu = PoincareDistance.grad(u, v, squnorm, sqvnorm, sqdist, ctx.eps)
        gv = PoincareDistance.grad(v, u, sqvnorm, squnorm, sqdist, ctx.eps)
        return g.expand_as(gu) * gu, g.expand_as(gv) * gv, None




class PoincareManifold(nn.Module):  #
    # Functions related to the Poincaré model
    '''
    Here, the curvature is removed, assuming beta=-c and c=1
    Ball radius 1/sqrt(c)
    '''

    def __init__(self, args, EPS=1e-8, PROJ_EPS=1e-10):
        super().__init__()
        self.args = args
        # self.logger = logger
        self.EPS = EPS
        self.eps = {torch.float32: 4e-3, torch.float64: 1e-5}
        self.PROJ_EPS = PROJ_EPS
        self.tanh = nn.Tanh()

    def normalize(self, x):
        return clip_by_norm(x, (1. - self.PROJ_EPS))

    def vec_vec_mobius_addition(self, u, v):
        # Input u and v are both [1, feature]
        # Output is [1, feature_dim]
        # norm_u_2 = (torch.linalg.norm(u,dim=1))**2
        # norm_v_2 = (torch.linalg.norm(v,dim=1)) ** 2
        norm_u_2 = torch.mm(v, v.t())  # New method for calculating the squared norm
        norm_v_2 = torch.mm(u, u.t())

        uv_dot_times = 2 * torch.dot(u[0], v[0])  # Is the coefficient here 2?
        denominator = 1 + uv_dot_times + norm_u_2 * norm_v_2  # This is the denominator

        coef_1 = (1 + uv_dot_times + norm_v_2) / (denominator + self.EPS)
        coef_2 = (1 - norm_u_2) / (denominator + self.EPS)
        res = coef_1 * u + coef_2 * v
        return res

    def matrix_matrix_mobius_addition(self, u, v):
        # Input u and v are both [node number, feature]
        # Output is the addition result between corresponding rows of u and v

        v = v
        norm_u_2 = torch.sum(torch.mul(u, u), dim=1)
        norm_v_2 = torch.sum(torch.mul(v, v), dim=1)

        uv_dot_times = 2 * torch.sum(torch.mul(u, v), dim=1)
        denominator = 1 + uv_dot_times + norm_u_2 * norm_v_2
        coef_1 = (1 + uv_dot_times + norm_v_2) / (denominator + self.EPS)
        coef_2 = (1 - norm_u_2) / (denominator + self.EPS)
        return (coef_1 * u.t() + coef_2 * v.t()).t()

    def mobius_matvec(self, m, x, c=1):
        '''
        Möbius matrix-vector multiplication,
        x: node_number, in_features
        m: out_features, in_features
        '''

        sqrt_c = c ** 0.5
        x_norm = x.norm(dim=-1, keepdim=True, p=2).clamp_min(self.EPS)
        mx = x @ m.transpose(-1, -2)
        mx_norm = mx.norm(dim=-1, keepdim=True, p=2).clamp_min(self.EPS)
        res_c = self.tanh(mx_norm / x_norm * th_atanh(sqrt_c * x_norm, self.EPS)) * mx / (mx_norm * sqrt_c)
        cond = (mx == 0).prod(-1, keepdim=True, dtype=torch.uint8)
        res_0 = torch.zeros(1, dtype=res_c.dtype, device=res_c.device)
        res = torch.where(cond, res_0, res_c)
        return res

    def vec_vec_distance(self, x, y):
        # Distance in the Poincaré model
        # x, y  [1, feature]

        norm_x_2 = torch.mm(x, x.t())
        norm_y_2 = torch.mm(y, y.t())
        norm_x_y_2 = torch.mm(x - y, (x - y).t())
        res = 1 + 2 * norm_x_y_2 / ((1 - norm_x_2 + self.EPS) * (1 - norm_y_2 + self.EPS))
        dis = torch.acosh(res)
        return dis

    def matrix_matrix_distance(self, x, y):
        # Distance in the Poincaré model
        '''
        The input consists of two matrices of the same shape x=n*n' and y=n*n',
        and it calculates the distance between corresponding rows.
        Expected output: n*1
        '''

        return PoincareDistance.apply(x, y, 1e-5)


    def matrix_self_matrix_distance(self, x, y):
        x = y

    def lambda_x(self, x):
        # return 2. / (1 - torch.mm(x, x.t()).item())
        return 2. / (1 - th_dot(x, x))

    def exp_map_zero(self, v,c=1):  # Mapping from the origin to the tangent space
        # Defines the input matrix, M*N, where N is the embedding dimension, and c is a parameter

        v = v + self.EPS
        norm_v = th_norm(v)  # [batch_size, 1]
        result = self.tanh(norm_v) / (norm_v) * v
        return self.normalize(result)

    def exp_map_x(self, input, x):  # Mapping from the origin to the tangent space
        """
        Exp map from tangent space of x to hyperbolic space
        """
        # Defines the input matrix, M*N, where N is the embedding dimension, and c is a parameter

        input = input
        lambda_x = self.lambda_x(x)
        norm_input = torch.linalg.norm(input, dim=-1, keepdim=True)
        y = torch.tanh(lambda_x * norm_input / 2)
        y = y * input / norm_input

        x = x.expand(input.shape[0], -1)
        res = self.matrix_matrix_mobius_addition(x, y)  # c=1
        return res

    def log_map_zero(self, y):  #
        # Can handle both 2D and 3D tensors
        # 2D input has dimensions M*N, where N is the embedding dimension, and c is a parameter

        diff = y + self.EPS
        norm_diff = th_norm(diff)

        return 1. / th_atanh(norm_diff, self.EPS) / norm_diff * diff

    def log_map_x(self, y, x):
        # y is a 2D tensor of shape M*n
        # x is a specific point of shape 1*n

        # diff = self.mob_add(-x, y)
        x_2 = x.expand(y.shape[0], -1)
        fu_x_y_add = self.matrix_matrix_mobius_addition(-x_2, y)
        norm_fu_x_y_add = torch.linalg.norm(fu_x_y_add, dim=-1, keepdim=True)
        res = 2 / self.lambda_x(x)
        res = res * torch.atanh(torch.clamp(norm_fu_x_y_add, min=-clip_value, max=clip_value)) * fu_x_y_add / (
                norm_fu_x_y_add + self.EPS)
        return res
    def mobius_add(self, x, y, c, dim=-1):
        x2 = x.pow(2).sum(dim=dim, keepdim=True)
        y2 = y.pow(2).sum(dim=dim, keepdim=True)
        xy = (x * y).sum(dim=dim, keepdim=True)
        num = (1 + 2 * c * xy + c * y2) * x + (1 - c * x2) * y
        denom = 1 + 2 * c * xy + c ** 2 * x2 * y2
        return num / denom.clamp_min(self.EPS)
    def proj(self, x, c=1):
        '''
        Ensures that the mapped points remain in the hyperbolic space.
        '''

        norm = torch.clamp_min(x.norm(dim=-1, keepdim=True, p=2), self.EPS)
        maxnorm = (1 - self.eps[x.dtype]) / (c ** 0.5)
        cond = norm > maxnorm
        projected = x / norm * maxnorm
        return torch.where(cond, projected, x)

    def proj_tan(self, u, p, c=1):
        '''
        Ensures that the mapped points remain in the tangent space, which is Euclidean space.
        '''
        return u

    def proj_tan0(self, u, c=1):
        return u


    def parallel_transport(self, src, dst, v):
        return self.lambda_x(src) / torch.clamp(self.lambda_x(dst), min=self.EPS) * self.gyr(dst, -src, v)

    def rgrad(self, p, d_p):
        """
        Function to compute Riemannian gradient from the
        Euclidean gradient in the Poincare ball.
        Args:
            p (Tensor): Current point in the ball
            d_p (Tensor): Euclidean gradient at p
        """
        p_sqnorm = torch.sum(p.data ** 2, dim=-1, keepdim=True)
        d_p = d_p * ((1 - p_sqnorm) ** 2 / 4.0).expand_as(d_p)
        return d_p

    def gyr(self, u, v, w):
        u_norm = th_dot(u, u)
        v_norm = th_dot(v, v)
        u_dot_w = th_dot(u, w)
        v_dot_w = th_dot(v, w)
        u_dot_v = th_dot(u, v)
        A = - u_dot_w * v_norm + v_dot_w + 2 * u_dot_v * v_dot_w
        B = - v_dot_w * u_norm - u_dot_w
        D = 1 + 2 * u_dot_v + u_norm * v_norm
        return w + 2 * (A * u + B * v) / (D + self.EPS)

    def metric_tensor(self, x, u, v):
        """
        The metric tensor in hyperbolic space.
        In-place operations for saving memory. (do not use this function in forward calls)
        """
        u_dot_v = th_dot(u, v)
        lambda_x = self.lambda_x(x)
        lambda_x *= lambda_x
        lambda_x *= u_dot_v
        return lambda_x
