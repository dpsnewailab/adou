import gc
import cv2
import torch
import pdf2image
import numpy as np
from nms import nms
from collections import OrderedDict
from torchvision.models.detection import MaskRCNN
from torchvision.transforms import functional as F
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from adou.image.network.EfficientNet import EfficientNet
from adou.image.network.FishNet import fishnet99, fishnet150, fishnet201

from adou.meta import Model, ModelType

class MaskRCNN(Model, metaclass=ModelType):
    """
    MaskRCNN model for Document Layout Analysis with different backbone:
        - ResNet
        - EfficientNet
        - FishNet
    """
    
    def __init__(self,
                    backbone=None,
                    architecture=None,
                    detector=None,
                    num_classes=None,
                    device='cpu',
                    *args, **kwargs):
        
        assert backbone is not None, ValueError('backbone can not None')
        assert architecture is not None, ValueError('architecture can not None')
        assert detector is not None, ValueError('detector can not None')
        assert num_classes is not None, ValueError('num_classes can not None')
        assert device is not None, ValueError('device can not None.')
        self.device = device

        super.__init__()
        if backbone == 'efficientnet':
            backbone = EfficientNet.from_pretrained(architecture)
            backbone.out_channels = 1280
        elif backbone == 'fishnet':
            if architecture == 'fishnet99':
                backbone = fishnet99()
            elif architecture == 'fishnet150':
                backbone = fishnet150()
            else:
                backbone = fishnet201()
            
            backbone.out_channels = 1000
        elif backbone == 'resnet':
            backbone = resnet_fpn_backbone(architecture, pretrained=True)
        
        self.model = MaskRCNN(backbone, num_classes=num_classes)
        self.model.to(device)
    
    def load(self, path=None, *args, **kwargs):
        assert path is not None, ValueError('path can not None.')

        if self.device == 'cuda':
            self.model.load_state_dict(torch.load(path))
        else:
            self.model.load_state_dict(torch.load(path, map_location=lambda storage, loc: storage))
            if 'nn_parallel_to_cpu' in kwargs:
                state_dict = torch.load(path, map_location=lambda storage, loc: storage)
                state_dict_without_nnparallel = OrderedDict()

                for key, item in state_dict.items():
                    state_without_nnparallel[key[7:]] = item

                self.model.load_state_dict(state_dict_without_nnparallel)

    def _analyze(self, img=None, *args, **kwargs):
        assert img is not None, ValueError('img can not be None')

        img = F.to_tensor(img)

        output = self.model([img])[0]
        for key, item in output.items():
            if self.device == 'cuda':
                item = item.cpu()
            output[key] = item.detach().numpy()

        boxes = [[x1, y1, x2 - x1, y2 - y1] for x1, y1, x2, y2 in output['boxes']]
        scores = output['scores']
        rects = nms.boxes(rects=boxes, scores=scores, nms_threshold=0.25)
        output['boxes'] = [output['boxes'][id] for id in rects if output['scores'][id] > 0.5]
        output['labels'] = [output['labels'][id] for id in rects if output['scores'][id] > 0.5]
        output['scores'] = [output['scores'][id] for id in rects if output['scores'][id] > 0.5]
        
        return output

    def batch_analyze(self, images=None, *args, **kwargs):
        """
        Analyze for a batch of images
        :param images:
        :return:
        """
        assert images is not None, ValueError('images can not be None')

        with torch.no_grad():
            if self.device == 'cuda':
                torch.cuda.synchronize()

            _images = []
            for image in images:
                _images.append(F.to_tensor(image).to(self.device))
                del image

            l_images = images.__len__()
            del images

            output = self.model(_images)
            _images = []
            del _images

            if 'use_listmemmap' in kwargs:
                f_out = ListMemMap()
            else:
                f_out = List()

            for id in range(output.__len__()):
                for key, item in output[id].items():
                    if self.device == 'cuda':
                        item = item.cpu()
                    output[id][key] = item.detach().numpy()
                del item

                boxes = [[x1, y1, x2 - x1, y2 - y1] for x1, y1, x2, y2 in output[id]['boxes']]
                scores = output[id]['scores']
                rects = nms.boxes(rects=boxes, scores=scores, nms_threshold=0.25)

                tmp = list()
                tmp.append([output[id]['boxes'][idx] for idx in rects if output[id]['scores'][idx] > 0.5])
                tmp.append([output[id]['labels'][idx] for idx in rects if output[id]['scores'][idx] > 0.5])
                tmp.append([output[id]['scores'][idx] for idx in rects if output[id]['scores'][idx] > 0.5])

                f_out.append(tmp)
                del tmp

            del output, l_images, boxes, scores, rects
            if self.device == 'cuda':
                torch.cuda.empty_cache()

            gc.collect()
            return f_out

    def analyze(self, img=None, *args, **kwargs):
        """
        :param img: PIL.Image
        :return:
        """
        assert img is not None, ValueError('img can not be None')

        with torch.no_grad():
            img = F.to_tensor(img)
            output = self.model([img])[0]
            for key, item in output.items():
                if self.device == 'cuda':
                    item = item.cpu()
                output[key] = item.detach().numpy()

            boxes = [[x1, y1, x2 - x1, y2 - y1] for x1, y1, x2, y2 in output['boxes']]
            scores = output['scores']
            rects = nms.boxes(rects=boxes, scores=scores, nms_threshold=0.25)
            output['boxes'] = [output['boxes'][id] for id in rects if output['scores'][id] > 0.5]
            output['labels'] = [output['labels'][id] for id in rects if output['scores'][id] > 0.5]
            output['scores'] = [output['scores'][id] for id in rects if output['scores'][id] > 0.5]
            del boxes, scores, rects, img

        return output

    def analyze_pdf(self, pdf_file=None, *args, **kwargs):
        assert pdf_file is not None, ValueError('pdf_file is not None')

        images = pdf2image.convert_from_path(pdf_file)
        for idx, image in enumerate(images):
            out = self.analyze(image)
            img = self.box_display(image, out)
            cv2.imwrite(os.path.join('test', 'show', pdf_file.split('/')[-1] + '_' + str(idx) + '.png'), img)

    def box_display(self, image, output, *args, **kwargs):
        assert image is not None, ValueError('image can not None')
        assert output is not None, ValueError('output can not None ')

        image = np.array(image)

        for idx, box in enumerate(output[0]):
            x1, y1, x2, y2 = box
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 20, 200), 10)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(image, '%s: %.2f' % (self.class_names[output[1][idx]], output[2][idx]),
                        (int(x1) + 10, int(y1) + 35),
                        font, 1, (0, 0, 255), 2, cv2.LINE_AA)
        return image

    def mask_display(self, image, output):
        assert image is not None, ValueError('image can not None')
        assert output is not None, ValueError('output can not None')

        image = np.array(image)

        masks = output['masks']
        _masks = masks.argmax(axis=0)
        _masks = np.reshape(_masks, (_masks.shape[1], _masks.shape[2]))

        for i in range(_masks.shape[0]):
            for j in range(_masks.shape[1]):
                if (_masks[i][j] > 0) and (_masks[i][j] < 8):
                    image[i][j] = self.colours[_masks[i][j]]

        return image