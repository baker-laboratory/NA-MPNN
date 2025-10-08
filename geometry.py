import torch
from torch.nn import functional as F

# ============================================================
def get_ang(a : torch.Tensor,
            b : torch.Tensor,
            c : torch.Tensor) -> torch.Tensor:
    """Calculate planar angles for three sets of points

    Args:
        a,b,c: xyz coordinates of the three sets of atoms, [...,3]
    Returns:
        planar angles (in rad) between a-b-c triples, [...]
    """

    v = F.normalize(a-b, dim=-1)
    w = F.normalize(c-b, dim=-1)
    #v = v / torch.norm(v, dim=-1, keepdim=True)
    #w = w / torch.norm(w, dim=-1, keepdim=True)

    y = torch.norm(v-w,dim=-1)
    x = torch.norm(v+w,dim=-1)
    ang = 2*torch.atan2(y, x)

    return ang


# ============================================================
def get_dih(a : torch.Tensor,
            b : torch.Tensor,
            c : torch.Tensor,
            d : torch.Tensor) -> torch.Tensor:
    """Calculate torsion angles from four sets of points

    Args:
        a,b,c,d: xyz coordinates of the four sets of atoms, [...,3]
    Returns:
        dihedral angles (in rad) formed by a-b-c-d quadruples, [...]
    """

    b0 = a - b
    b1 = c - b
    b2 = d - c

    #b1 = b1 / torch.norm(b1, dim=-1, keepdim=True)
    b1 = F.normalize(b1, dim=-1)

    v = b0 - torch.sum(b0*b1, dim=-1, keepdim=True)*b1
    w = b2 - torch.sum(b2*b1, dim=-1, keepdim=True)*b1

    x = torch.sum(v*w, dim=-1)
    y = torch.sum(torch.cross(b1,v,dim=-1)*w, dim=-1)

    return torch.atan2(y, x)


# ============================================================
def get_frames(a : torch.Tensor,
               b : torch.Tensor,
               c : torch.Tensor) -> torch.Tensor:
    """Build local frames from three sets of points

    Args:
        a,b,c: xyz coordinates of the three sets of atoms, [...,3]
    Returns:
        orthogonal frame of reference, [...,3,3]
        ex=[...,:,0] - (a->b) x (c->b), perpendicular to the abc plane
        ey=[...,:,1] - ez x ex
        ez=[...,:,2] - (a->b) + (c->b), in the abc plane
    """

    x = b-a
    y = c-b

    e3 = x-y
    e1 = torch.cross(y,x, dim=-1)
    e2 = torch.cross(e3,e1, dim=-1)

    u = torch.stack((e1,e2,e3), dim=-1)
    u = F.normalize(u, dim=-2)

    return u


# ============================================================
def triple_prod(a : torch.Tensor,
                b : torch.Tensor,
                c : torch.Tensor,
                norm : bool = False) -> torch.Tensor:
    '''Get triple product of three sets of vectors

    Args:
        a,b,c: xyz coordinates of the three sets of atoms, [...,3]
        norm: bool flag to indicate whether the input vectors should be
              unit-normalized before computing the product
    Returns:
        a.dot(b.cross(c)), [...,3]
    '''
    
    if norm==True:
        # normalize vectors before computing the product
        bxc = torch.cross(input = F.normalize(b, dim=-1), #b/b.norm(dim=-1,keepdim=True),
                          other = F.normalize(c, dim=-1), #c/c.norm(dim=-1,keepdim=True),
                          dim = -1)
        abc = (F.normalize(a,dim=-1)*bxc).sum(-1)
    else:
        # compute as is
        abc = (a*torch.cross(b,c,dim=-1)).sum(-1)
    
    return abc
