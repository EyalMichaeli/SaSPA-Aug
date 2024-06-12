import torch
import torch.nn.functional as F
import torch.nn as nn


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.token_embedding = clip_model.token_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, tokenized_prompts):
        x = self.token_embedding(tokenized_prompts).type(self.dtype)

        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x
    

class CLIP_selector(nn.Module):
    """
    taken from the CLIP filtering from https://github.com/CVMI-Lab/SyntheticData
    """
    def __init__(self, clip_model, train_preprocess, val_preprocess, tokenized_prompts):
        super().__init__()
        # self.prompt_learner = PromptLearner(args, classnames, clip_model)
        # self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.train_preprocess = train_preprocess
        self.val_preprocess = val_preprocess
        self.tokenized_prompts = tokenized_prompts

    def forward(self, image):
        with torch.no_grad():
            image_features = self.image_encoder(image.type(self.dtype))

            # prompts = self.prompt_learner()
            tokenized_prompts = self.tokenized_prompts
            text_features = self.text_encoder(tokenized_prompts)

            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            logit_scale = self.logit_scale.exp()
            logits = logit_scale * image_features @ text_features.t()

        return logits
    


class SoftTargetCrossEntropy_T(torch.nn.Module):
    '''
    from timm, abandon
    taken from: https://github.com/CVMI-Lab/SyntheticData/blob/main/src/classifier-tuning/src/models/utils.py
    '''

    def __init__(self, T=2.0):  # T: temperature. =2.0 in paper and original repo.
        super(SoftTargetCrossEntropy_T, self).__init__()
        self.T = T

    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        '''
        Args:
            x: student logit
            target: teacher logit
        note that target is a logit as well (should be by CLIP or other model)
        Returns: 

        '''
        soft_labels = torch.softmax(target/self.T, dim=1)
        loss = torch.sum(-soft_labels * F.log_softmax(x, dim=-1), dim=-1)
        # loss = torch.sum(-target * F.log_softmax(x, dim=-1), dim=-1)
        return loss.mean()
    