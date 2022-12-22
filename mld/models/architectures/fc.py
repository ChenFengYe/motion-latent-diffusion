import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder_FC(nn.Module):
    def __init__(self, modeltype, njoints, nfeats, num_frames, num_classes, translation, pose_rep, glob, glob_rot,
                 latent_dim=256, **kargs):
        super().__init__()

        self.modeltype = modeltype
        self.njoints = njoints
        self.nfeats = nfeats
        self.num_frames = num_frames
        self.num_classes = num_classes
        self.translation = translation
        self.pose_rep = pose_rep
        self.glob = glob
        self.glob_rot = glob_rot

        self.latent_dim = latent_dim

        self.activation = nn.GELU()

        self.input_dim = self.njoints*self.nfeats*self.num_frames+self.num_classes

        self.fully_connected = nn.Sequential(nn.Linear(self.input_dim, 512),
                                             nn.GELU(),
                                             nn.Linear(512, 256),
                                             nn.GELU())
        if self.modeltype == "cvae":
            self.mu = nn.Linear(256, self.latent_dim)
            self.var = nn.Linear(256, self.latent_dim)
        else:
            self.final = nn.Linear(256, self.latent_dim)

    def forward(self, batch):
        x, y = batch["x"], batch["y"]
        bs, njoints, feats, nframes = x.size()
        if (njoints * feats * nframes) != self.njoints*self.nfeats*self.num_frames:
            raise ValueError("This model is not adapted with this input")
        
        if len(y.shape) == 1:  # can give on hot encoded as input
            y = F.one_hot(y, self.num_classes)
        y = y.to(dtype=x.dtype)
        x = x.reshape(bs, njoints*feats*nframes)
        x = torch.cat((x, y), 1)

        x = self.fully_connected(x)

        if self.modeltype == "cvae":
            return {"mu": self.mu(x), "logvar": self.var(x)}
        else:
            return {"z": self.final(x)}


class Decoder_FC(nn.Module):
    def __init__(self, modeltype, njoints, nfeats, num_frames, num_classes, translation, pose_rep, glob, glob_rot,
                 latent_dim=256, **kargs):
        super().__init__()

        self.modeltype = modeltype
        self.njoints = njoints
        self.nfeats = nfeats
        self.num_frames = num_frames
        self.num_classes = num_classes
        self.translation = translation
        self.pose_rep = pose_rep
        self.glob = glob
        self.glob_rot = glob_rot

        self.latent_dim = latent_dim

        self.input_dim = self.latent_dim + self.num_classes
        self.output_dim = self.njoints*self.nfeats*self.num_frames

        self.fully_connected = nn.Sequential(nn.Linear(self.input_dim, 256),
                                             nn.GELU(),
                                             nn.Linear(256, 512),
                                             nn.GELU(),
                                             nn.Linear(512, self.output_dim),
                                             nn.GELU())
        
    def forward(self, batch):
        z, y = batch["z"], batch["y"]
        # z: [batch_size, latent_dim]
        # y: [batch_size]
        if len(y.shape) == 1:  # can give on hot encoded as input
            y = F.one_hot(y, self.num_classes)
        y = y.to(dtype=z.dtype)  # y: [batch_size, num_classes]
        # z: [batch_size, latent_dim+num_classes]
        z = torch.cat((z, y), dim=1)
        
        z = self.fully_connected(z)

        bs, _ = z.size()

        z = z.reshape(bs, self.njoints, self.nfeats, self.num_frames)
        batch["output"] = z
        return batch
