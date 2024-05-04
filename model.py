import torch
import numpy as np
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
from torch.autograd import Variable

def model_handler(model, class_num):
    if model == 'resnet18':
        return resnet18(class_num=class_num)
    elif model == 'vgg11':
        return vgg11(class_num=class_num)
    elif model == 'me_resnet18':
        return me_resnet18(class_num=class_num)
    elif model == 'me_vgg11':
        return me_vgg11(class_num=class_num)
    elif model == 'mfd_resnet18':
        return mfd_resnet18(class_num=class_num)
    elif model == 'mfd_vgg11':
        return mfd_vgg11(class_num=class_num)
    elif model == 'me_mfd_resnet18':
        return me_mfd_resnet18(class_num=class_num)
    elif model == 'me_mfd_vgg11':
        return me_mfd_vgg11(class_num=class_num)
    elif model == 'hsic_resnet18':
        return hsic_resnet18(class_num=class_num)
    elif model == 'hsic_vgg11':
        return hsic_vgg11(class_num=class_num)
    elif model == 'me_hsic_resnet18':
        return me_hsic_resnet18(class_num=class_num)
    elif model == 'me_hsic_vgg11':
        return me_hsic_vgg11(class_num=class_num)
    else:
        return None


def forward_features(model, x):
    """
    Dump each IC's feature map.
    Args:
        model (nn.module) : model
        x (torch.tensor) : model input
    """
    early_exits_outputs = []
    for layer, early_exits_layer in model.f:
        x = layer(x)
        if early_exits_layer != None:
            ic_out = early_exits_layer[0](x)
            ic_out = early_exits_layer[1](ic_out)
            ic_out = early_exits_layer[2](ic_out)
            ic_out = early_exits_layer[3](ic_out)
            ic_out = early_exits_layer[4](ic_out)
            early_exits_outputs.append(ic_out)
    final_out = x
    early_exits_outputs.append(final_out) # append final out
    return early_exits_outputs

class resnet18(nn.Module):
    def __init__(self, class_num=8, pretrained=True):
        super(resnet18, self).__init__()
        self.model = models.resnet18(pretrained=pretrained)
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, class_num)

    def forward(self, x):
        return self.model(x)
        
class vgg11(nn.Module):
    def __init__(self, class_num=8, pretrained=True):
        super(vgg11, self).__init__()
        self.model = models.vgg11_bn()
        in_features = self.model.classifier[-1].in_features
        self.model.classifier[-1] = nn.Linear(in_features, class_num)

    def forward(self, x):
        return self.model(x)

class me_resnet18(nn.Module):
    def __init__(self, pretrained=True, class_num=8):
        super(me_resnet18, self).__init__()
        self.f = nn.ModuleList()
        self.num_output = 5
        self.confidence_threshold = 0.8

        for name, module in models.resnet18(pretrained=pretrained).named_children():
            if isinstance(module, nn.Linear):
                continue
    
            if isinstance(module, nn.Sequential):
                exit_branch = nn.Sequential(nn.Conv2d(self.get_out_channels(module), 64, kernel_size=3, stride=1, padding=0, bias=True),
                                            nn.AdaptiveAvgPool2d(output_size=(64)),
                                            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=0, bias=True),
                                            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
                                            nn.Flatten(),
                                            nn.Linear(32, class_num, bias=True)
                                           )
                self.f.append(nn.ModuleList([module, exit_branch]))
            else:
                self.f.append(nn.ModuleList([module, None]))
        self.g = nn.Linear(512, class_num, bias=True)
        
    def get_out_channels(self, module):
        for name, out_module in module.named_modules():
            if name == '1.bn2':
                return out_module.num_features

    def forward(self, x):
        early_exits_outputs = []
        for layer, early_exits_layer in self.f:
            x = layer(x)
            if early_exits_layer != None:
                early_exits_outputs.append(early_exits_layer(x))
        final_out = self.g(torch.flatten(x, start_dim=1))
        early_exits_outputs.append(final_out) # append final out
        return early_exits_outputs
    
    def early_exit(self, x):
        outputs = []
        confidences = []
        output_id = 0
        for layer, early_exits_layer in self.f:
            x = layer(x)
            if early_exits_layer != None:
                early_exit_out = early_exits_layer(x)
                outputs.append(early_exit_out)
                softmax = nn.functional.softmax(early_exit_out[0], dim=0)
                confidence = torch.max(softmax).cpu().numpy()
                confidences.append(confidence)
                if confidence >= self.confidence_threshold:
                    is_early = True
                    return early_exit_out, output_id, is_early
                output_id += 1
                
        output = self.g(torch.flatten(x, start_dim=1))
        outputs.append(output)
        softmax = nn.functional.softmax(output[0], dim=0)
        confidence = torch.max(softmax).cpu().numpy()
        confidences.append(confidence)
        max_confidence_output = np.argmax(confidences)
        is_early = False
        return outputs[max_confidence_output], max_confidence_output, is_early

class me_vgg11(nn.Module):
    def __init__(self, class_num=8, pretrained=True):
        super(me_vgg11, self).__init__()
        self.f = nn.ModuleList()
        self.num_output = 5
        self.confidence_threshold = 0.8
        exit_branch_pos = [5, 10, 15, 20]
        num_channel = {
                       '5':128, 
                       '10':256,
                       '15':512,
                       '20':512
                      }
        # self.model = models.vgg11(pretrained=pretrained)

        for name, module in models.vgg11(pretrained=True).features.named_children():
            if int(name) in exit_branch_pos:
                exit_branch = nn.Sequential(nn.Conv2d(num_channel[name], 64, kernel_size=3, stride=1, padding=0, bias=True),
                                                    nn.AdaptiveAvgPool2d(output_size=(64)),
                                                    nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=0, bias=True),
                                                    nn.AdaptiveAvgPool2d(output_size=(1, 1)),
                                                    nn.Flatten(),
                                                    nn.Linear(32, class_num, bias=True)
                                                   )
                self.f.append(nn.ModuleList([module, exit_branch]))
            else:
                self.f.append(nn.ModuleList([module, None]))
                
        self.f.append(nn.ModuleList([nn.AdaptiveAvgPool2d(output_size=(7, 7)), None]))
        self.f.append(nn.ModuleList([nn.Flatten(), None]))
            
        for name, module in models.vgg11(pretrained=True).classifier.named_children():
            if name != '6': #if not the last layer
                self.f.append(nn.ModuleList([module, None]))
            else:           #for last layer
                final_layer = nn.Linear(4096, class_num, bias=True)
                self.g = nn.Linear(4096, class_num, bias=True)

    def forward(self, x):
        output = []
        early_exits_outputs = []
        for layer, early_exits_layer in self.f:
            torch.cuda.empty_cache()
            x = layer(x)
            output.append(x)
            if early_exits_layer != None:
                early_exits_outputs.append(early_exits_layer(x))
        final_out = self.g(torch.flatten(x, start_dim=1))
        output.append(final_out)
        early_exits_outputs.append(final_out)
        return early_exits_outputs


class mfd_resnet18(nn.Module):
    def __init__(self, class_num=8, pretrained=True):
        super(mfd_resnet18, self).__init__()
        self.f = nn.ModuleList()
        self.model = models.resnet18(pretrained=pretrained)
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, class_num)
        for _, module in models.resnet18(pretrained=pretrained).named_children():
            if isinstance(module, nn.Linear):
                continue
            self.f.append(module)
        self.g = nn.Linear(512, class_num, bias=True)

    def forward(self, x):
        for layer in self.f:
            x = layer(x)
        final_out = self.g(torch.flatten(x, start_dim=1))

        return final_out, torch.flatten(x, start_dim=1)

class mfd_vgg11(nn.Module):
    def __init__(self, class_num=8, pretrained=True):
        super(mfd_vgg11, self).__init__()
        self.f = nn.ModuleList()
        self.g = None
        self.model = models.vgg11(pretrained=pretrained)

        for name, module in models.vgg11(pretrained=True).features.named_children():
            self.f.append(nn.ModuleList([module, None]))
                
        # self.f.append(nn.ModuleList([nn.Flatten(), None])) # Flatten the features
        self.f.append(nn.ModuleList([nn.AdaptiveAvgPool2d(output_size=(7, 7)), None]))
        self.f.append(nn.ModuleList([nn.Flatten(), None])) # Flatten the features
            
        for name, module in models.vgg11(pretrained=True).classifier.named_children():
            if name != '6': #if not the last layer
                self.f.append(nn.ModuleList([module, None]))
            else:           #for last layer
                final_layer = nn.Linear(4096, class_num, bias=True)
                self.g = nn.Linear(4096, class_num, bias=True)

    def forward(self, x):
        for layer, early_exits_layer in self.f:
            torch.cuda.empty_cache()
            x = layer(x)
        final_out = self.g(torch.flatten(x, start_dim=1))
        return final_out, torch.flatten(x, start_dim=1)

class me_mfd_resnet18(nn.Module):
    def __init__(self, pretrained=True, class_num=8):
        super(me_mfd_resnet18, self).__init__()
        self.f = nn.ModuleList()
        self.num_output = 5
        self.confidence_threshold = 0.8

        for name, module in models.resnet18(pretrained=pretrained).named_children():
            if isinstance(module, nn.Linear):
                continue
    
            if isinstance(module, nn.Sequential):
                exit_branch = nn.Sequential(nn.Conv2d(self.get_out_channels(module), 64, kernel_size=3, stride=1, padding=0, bias=True),
                                            nn.AdaptiveAvgPool2d(output_size=(64)),
                                            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=0, bias=True),
                                            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
                                            nn.Flatten(),
                                            nn.Linear(32, class_num, bias=True)
                                           )
                self.f.append(nn.ModuleList([module, exit_branch]))
            else:
                self.f.append(nn.ModuleList([module, None]))
        self.g = nn.Linear(512, class_num, bias=True)
        
    def get_out_channels(self, module):
        for name, out_module in module.named_modules():
            if name == '1.bn2':
                return out_module.num_features

    def forward(self, x):
        early_exits_outputs = []
        for layer, early_exits_layer in self.f:
            x = layer(x)
            if early_exits_layer != None:
                early_exits_outputs.append(early_exits_layer(x))
        final_out = self.g(torch.flatten(x, start_dim=1))
        early_exits_outputs.append(final_out) # append final out
        return early_exits_outputs

class me_mfd_vgg11(nn.Module):
    def __init__(self, class_num=8, pretrained=True):
        super(me_mfd_vgg11, self).__init__()
        self.f = nn.ModuleList()
        self.num_output = 5
        self.confidence_threshold = 0.8
        exit_branch_pos = [5, 10, 15, 20]
        num_channel = {
                       '5':128, 
                       '10':256,
                       '15':512,
                       '20':512
                      }
        # self.model = models.vgg11(pretrained=pretrained)

        for name, module in models.vgg11(pretrained=True).features.named_children():
            if int(name) in exit_branch_pos:
                exit_branch = nn.Sequential(nn.Conv2d(num_channel[name], 64, kernel_size=3, stride=1, padding=0, bias=True),
                                                    nn.AdaptiveAvgPool2d(output_size=(64)),
                                                    nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=0, bias=True),
                                                    nn.AdaptiveAvgPool2d(output_size=(1, 1)),
                                                    nn.Flatten(),
                                                    nn.Linear(32, class_num, bias=True)
                                                   )
                self.f.append(nn.ModuleList([module, exit_branch]))
            else:
                self.f.append(nn.ModuleList([module, None]))
                
        self.f.append(nn.ModuleList([nn.AdaptiveAvgPool2d(output_size=(7, 7)), None]))
        self.f.append(nn.ModuleList([nn.Flatten(), None]))
            
        for name, module in models.vgg11(pretrained=True).classifier.named_children():
            if name != '6': #if not the last layer
                self.f.append(nn.ModuleList([module, None]))
            else:           #for last layer
                final_layer = nn.Linear(4096, class_num, bias=True)
                self.g = nn.Linear(4096, class_num, bias=True)

    def forward(self, x):
        output = []
        early_exits_outputs = []
        for layer, early_exits_layer in self.f:
            torch.cuda.empty_cache()
            x = layer(x)
            output.append(x)
            if early_exits_layer != None:
                early_exits_outputs.append(early_exits_layer(x))
        final_out = self.g(torch.flatten(x, start_dim=1))
        output.append(final_out)
        early_exits_outputs.append(final_out)
        return early_exits_outputs

class hsic_resnet18(nn.Module):
    def __init__(self, pretrained=True, class_num=8):
        super(hsic_resnet18, self).__init__()
        self.f = nn.ModuleList()
        self.g = None
        for name, module in models.resnet18(pretrained=pretrained).named_children():
            if isinstance(module, nn.Linear):
                continue
            self.f.append(nn.ModuleList([module, None]))
            
        self.g = nn.Linear(512, class_num, bias=True)

    def forward(self, x):
        output = []
        for layer, none_layer in self.f:
            torch.cuda.empty_cache()
            x = layer(x)
            output.append(x)
        final_out = self.g(torch.flatten(x, start_dim=1))
        output.append(final_out)
        return output
    
class hsic_vgg11(nn.Module):
    def __init__(self, pretrained=True, class_num=114):
        super(hsic_vgg11, self).__init__()
        self.f = nn.ModuleList()
        # self.model = models.vgg11(pretrained=pretrained)
        self.g = None
        for name, module in models.vgg11(pretrained=True).features.named_children():
            self.f.append(nn.ModuleList([module, None]))
                
        self.f.append(nn.ModuleList([nn.AdaptiveAvgPool2d(output_size=(7, 7)), None]))
        self.f.append(nn.ModuleList([nn.Flatten(), None]))
            
        for name, module in models.vgg11(pretrained=True).classifier.named_children():
            if name != '6': #if not the last layer
                self.f.append(nn.ModuleList([module, None]))
            else:           #for last layer
                self.g = nn.Linear(4096, class_num, bias=True)

    def forward(self, x):
        output = []
        for layer, none_layer in self.f:
            torch.cuda.empty_cache()
            x = layer(x)
            output.append(x)
        final_out = self.g(torch.flatten(x, start_dim=1))
        output.append(final_out)
        return output

class me_hsic_resnet18(nn.Module):
    def __init__(self, pretrained=True, class_num=8):
        super(me_hsic_resnet18, self).__init__()
        self.f = nn.ModuleList()
        self.num_output = 5
        self.confidence_threshold = 0.8

        for name, module in models.resnet18(pretrained=pretrained).named_children():
            if isinstance(module, nn.Linear):
                continue
    
            if isinstance(module, nn.Sequential):
                exit_branch = nn.Sequential(nn.Conv2d(self.get_out_channels(module), 64, kernel_size=3, stride=1, padding=0, bias=True),
                                            nn.AdaptiveAvgPool2d(output_size=(64)),
                                            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=0, bias=True),
                                            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
                                            nn.Flatten(),
                                            nn.Linear(32, class_num, bias=True)
                                           )
                self.f.append(nn.ModuleList([module, exit_branch]))
            else:
                self.f.append(nn.ModuleList([module, None]))
        self.g = nn.Linear(512, class_num, bias=True)
        
    def get_out_channels(self, module):
        for name, out_module in module.named_modules():
            if name == '1.bn2':
                return out_module.num_features

    def forward(self, x):
        output = []
        early_exits_outputs = []
        for layer, early_exits_layer in self.f:
            x = layer(x)
            output.append(x)
            if early_exits_layer != None:
                early_exits_outputs.append(early_exits_layer(x))
        final_out = self.g(torch.flatten(x, start_dim=1))
        output.append(final_out)
        early_exits_outputs.append(final_out) # append final out
        return early_exits_outputs
    
class me_hsic_vgg11(nn.Module):
    def __init__(self, class_num=114, pretrained=True):
        super(me_hsic_vgg11, self).__init__()
        self.f = nn.ModuleList()
        self.num_output = 5
        self.confidence_threshold = 0.8
        exit_branch_pos = [5, 10, 15, 20]
        num_channel = {
                       '5':128, 
                       '10':256,
                       '15':512,
                       '20':512
                      }
        # self.model = models.vgg11(pretrained=pretrained)

        for name, module in models.vgg11(pretrained=True).features.named_children():
            if int(name) in exit_branch_pos:
                exit_branch = nn.Sequential(nn.Conv2d(num_channel[name], 64, kernel_size=3, stride=1, padding=0, bias=True),
                                                    nn.AdaptiveAvgPool2d(output_size=(64)),
                                                    nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=0, bias=True),
                                                    nn.AdaptiveAvgPool2d(output_size=(1, 1)),
                                                    nn.Flatten(),
                                                    nn.Linear(32, class_num, bias=True)
                                                   )
                self.f.append(nn.ModuleList([module, exit_branch]))
            else:
                self.f.append(nn.ModuleList([module, None]))
                
        self.f.append(nn.ModuleList([nn.AdaptiveAvgPool2d(output_size=(7, 7)), None]))
        self.f.append(nn.ModuleList([nn.Flatten(), None]))
            
        for name, module in models.vgg11(pretrained=True).classifier.named_children():
            if name != '6': #if not the last layer
                self.f.append(nn.ModuleList([module, None]))
            else:           #for last layer
                final_layer = nn.Linear(4096, class_num, bias=True)
                self.g = nn.Linear(4096, class_num, bias=True)

    def forward(self, x):
        output = []
        early_exits_outputs = []
        for layer, early_exits_layer in self.f:
            torch.cuda.empty_cache()
            x = layer(x)
            output.append(x)
            if early_exits_layer != None:
                early_exits_outputs.append(early_exits_layer(x))
        final_out = self.g(torch.flatten(x, start_dim=1))
        output.append(final_out)
        early_exits_outputs.append(final_out)
        return early_exits_outputs



    

