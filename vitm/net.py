import torch
import torch.nn as nn

from . import modules


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class ViTM(nn.Module):
    """ Main class which modules and calls them as part of the reasoning chain sequence.
    """
    def __init__(self,
                 vocab,
                 feature_dim=(1024, 14, 14),
                 module_dim=128,
                 cls_proj_dim=512,
                 fc_dim=1024):

        super().__init__()

        # The stem takes features from ResNet (or another feature extractor) and downsamples to appropriate size for modules
        self.stem = nn.Sequential(nn.Conv2d(feature_dim[0], module_dim, kernel_size=3, padding=1),
                                  nn.ReLU(),
                                  nn.Conv2d(module_dim, module_dim, kernel_size=3, padding=1),
                                  nn.ReLU()
                                 )

        module_rows, module_cols = feature_dim[1], feature_dim[2]

        # The classifier takes the output of the last module (which will be a Query or Equal module)
        # and produces a distribution over answers
        self.classifier = nn.Sequential(nn.Conv2d(module_dim, cls_proj_dim, kernel_size=1),
                                        nn.ReLU(inplace=True),
                                        nn.MaxPool2d(kernel_size=2, stride=2),
                                        Flatten(),
                                        nn.Linear(cls_proj_dim * module_rows * module_cols // 4,
                                                  fc_dim),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(fc_dim, 28)  # note no softmax here
                                       )

        self.function_modules = {}  # holds our modules
        self.vocab = vocab
        # go through the vocab and add all the modules to our model
        for module_name in vocab['program_token_to_idx']:
            if module_name in ['<NULL>', '<START>', '<END>', '<UNK>', 'unique']:
                continue  # we don't need modules for the placeholders
            
            # figure out which module we want we use
            if module_name == 'scene':
                # scene is just a flag that indicates the start of a new line of reasoning
                # we set `module` to `None` because we still need the flag 'scene' in forward()
                module = None
            elif module_name == 'intersect':
                module = modules.AndModule()
            elif module_name == 'union':
                module = modules.OrModule()
            elif 'equal' in module_name or module_name in {'less_than', 'greater_than'}:
                module = modules.ComparisonModule(module_dim)
            elif 'query' in module_name or module_name in {'exist', 'count'}:
                module = modules.QueryModule(module_dim)
            elif 'relate' in module_name:
                module = modules.RelateModule(module_dim)
            elif 'same' in module_name:
                module = modules.SameModule(module_dim)
            else:
                module = modules.AttentionModule(module_dim)

            # add the module to our dictionary and register its parameters so it can learn
            self.function_modules[module_name] = module
            self.add_module(module_name, module)

        # this is used as input to the first AttentionModule in each program
        ones = torch.ones(1, 1, module_rows, module_cols)
        self.ones_var = ones.cuda() if torch.cuda.is_available() else ones
        
        self._attention_sum = 0

    @property
    def attention_sum(self):
        return self._attention_sum

    def forward(self, feats, programs):
        batch_size = feats.size(0)
        assert batch_size == len(programs)

        feat_input_volume = self.stem(feats)  # forward all the features through the stem at once

        # We compose each module network individually since they are constructed on a per-question
        # basis. Here we go through each program in the batch, construct a modular network based on
        # it, and send the image forward through the modular structure. We keep the output of the
        # last module for each program in final_module_outputs. These are needed to then compute a
        # distribution over answers for all the questions as a batch.
        final_module_outputs = []
        self._attention_sum = 0
        for n in range(batch_size):
            feat_input = feat_input_volume[n:n+1] 
            output = feat_input
            saved_output = None
            for i in reversed(programs.data[n].cpu().numpy()):
                module_type = self.vocab['program_idx_to_token'][i]
                if module_type in {'<NULL>', '<START>', '<END>', '<UNK>', 'unique'}:
                    continue  # the above are no-ops in our model
                
                module = self.function_modules[module_type]
                if module_type == 'scene':
                    # store the previous output; it will be needed later
                    # scene is just a flag, performing no computation
                    saved_output = output
                    output = self.ones_var
                    continue
                
                if 'equal' in module_type or module_type in {'intersect', 'union', 'less_than',
                                                             'greater_than'}:
                    output = module(output, saved_output)  # these modules take two feature maps
                else:
                    # these modules take extracted image features and a previous attention
                    output = module(feat_input, output)

                if any(t in module_type for t in ['filter', 'relate', 'same']):
                    self._attention_sum += output.sum()
                    
            final_module_outputs.append(output)
            
        final_module_outputs = torch.cat(final_module_outputs, 0)
        return self.classifier(final_module_outputs)

    def forward_and_return_intermediates(self, program_var, feats_var):
        """ Forward program `program_var` and image features `feats_var` through ViTM
        and return an answer and intermediate outputs.
        """
        intermediaries = []
        # the logic here is the same as self.forward()
        scene_input = self.stem(feats_var)
        output = scene_input
        saved_output = None
        for i in reversed(program_var.data.cpu().numpy()[0]):
            module_type = self.vocab['program_idx_to_token'][i]
            if module_type in {'<NULL>', '<START>', '<END>', '<UNK>', 'unique'}:
                continue

            module = self.function_modules[module_type]
            if module_type == 'scene':
                saved_output = output
                output = self.ones_var
                intermediaries.append(None) # indicates a break/start of a new logic chain
                continue

            if 'equal' in module_type or module_type in {'intersect', 'union', 'less_than',
                                                         'greater_than'}:
                output = module(output, saved_output)
            else:
                output = module(scene_input, output)

            if module_type in {'intersect', 'union'}:
                intermediaries.append(None) # this is the start of a new logic chain

            if module_type in {'intersect', 'union'} or any(s in module_type for s in ['same',
                                                                                       'filter',
                                                                                       'relate']):
                intermediaries.append((module_type, output.data.cpu().numpy().squeeze()))

        _, pred = self.classifier(output).max(1)
        return (self.vocab['answer_idx_to_token'][pred.item()], intermediaries)


def load_vitm_net(checkpoint, vocab):
    """ Convenience function to load a ViTM model from a checkpoint file.
    """
    net = ViTM(vocab)
    net.load_state_dict(torch.load(str(checkpoint), map_location={'cuda:0': 'cpu'}))
    if torch.cuda.is_available():
        net.cuda()
    return net
