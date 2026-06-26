from collections import Counter

from torch.nn import functional as F
import torch.nn as nn
import numpy as np
import logging
import torch

from PIL import Image
from torchvision import transforms

class EmbeddingClassifier:
    def __init__(self, model_path, data_set_path, device='cpu', min_threshold=0.03, max_threshold=0.4):
        self.device = device
    
        self.softmax = nn.Softmax(dim=1)
        
        self.model = torch.jit.load(model_path)
        self.model.eval()
        self.model.to(device)
        
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold
        
        self.loader = transforms.Compose([
            transforms.Resize((224, 224), Image.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        data = torch.load(data_set_path)
        assert len(data) == 6
        
        self.data_base = data[0].to(device)
        self.top_k = 15
    
        self.internal_ids = data[1]
        self.image_ids = data[2]
        self.annotation_ids = data[3]
        self.drawn_fish_ids = data[4]
        self.keys = data[5]
        self.keys = {int(key): self.keys[key] for key in self.keys}
        
        logging.info("[INIT][CLASSIFICATION] Initialization of classifier was finished")
                
    def __inference(self, image): 
        logging.info("[PROCESSING][CLASSIFICATION] Getting embedding for a single detection mask")
        
        dump_embed, fc_output = self.model(image.unsqueeze(0).to(self.device))
        
        logging.info("[PROCESSING][CLASSIFICATION] Classification by Full Connected layer for a single detection mask")  
        classes, _ = self.__classify_fc(fc_output)
        
        fc_recognized = self.keys[classes[0].item()]
        
        logging.info("[PROCESSING][CLASSIFICATION] Classification by embedding for a single detection mask")
        output_by_embeddings = self.__classify_embedding(dump_embed[0])
        
        logging.info("[PROCESSING][CLASSIFICATION] Beautify output for a single detection mask")
        result = self.__beautify_output(output_by_embeddings, fc_recognized)
        return result
                        
    def __beautify_output(self, output_by_embeddings, classification_label):
        # Create a dictionary to track unique names
        unique_names = set()
        dict_results = []
        
        names = Counter([class_map['name'] for class_map in output_by_embeddings])

        # Add unique items from output_by_embeddings to dict_results
        for class_map in output_by_embeddings:
            name = class_map['name']
            if name not in unique_names:
                class_map['accuracy'] = names[name]/self.top_k
                class_map['times'] = names[name]
                
                dict_results.append(class_map)
                unique_names.add(name)

        # Add the classification label if it is not already present
        label_name = classification_label['label']
        if label_name not in unique_names:
            logging.info("[PROCESSING][CLASSIFICATION] Append into output classification result by FC - layer")
            dict_results.append({
                'name': label_name,
                'species_id': classification_label['species_id'],
                'image_id': None,
                'accuracy': 0.01,
                'times': 0,
                'annotation_id': None,
                'drawn_fish_id': None,
            })

        # Sort the results by accuracy in descending order
        dict_results.sort(key=lambda x: x['accuracy'], reverse=True)

        return dict_results

    def calculate_probability(self, distance):
        """
        Recalculates the classification probability based on the cosine distance.

        Arguments:
        distance (float): Cosine distance.

        Return:
        float: Recalculated classification probability.
        """
        if distance <= self.min_threshold:
            return 1.0
        elif distance >= self.max_threshold:
            return 0.0
        else:
            # Linear probability conversion
            return (self.max_threshold - distance) / (self.max_threshold - self.min_threshold)

    def inference_numpy(self, img):
        image = Image.fromarray(img)
        image = self.loader(image)
        
        return self.__inference(image)
    
    def batch_inference(self, imgs):
        batch_input = []
        for idx in range(len(imgs)):
            image = Image.fromarray(imgs[idx])
            image = self.loader(image)
            batch_input.append(image)

        batch_input = torch.stack(batch_input)
        dump_embeds, class_ids = self.model(batch_input)
        
        logging.info("[PROCESSING][CLASSIFICATION] Classification by Full Connected layer for a single detection mask")  
        classes, scores = self.__classify_fc(class_ids)
       
        outputs = []
        for output_id in range(len(classes)):

            logging.info("[PROCESSING][CLASSIFICATION] Classification by embedding for a single detection mask")
            output_by_embeddings = self.__classify_embedding(dump_embeds[output_id])
            result = self.__beautify_output(output_by_embeddings, self.keys[classes[output_id].item()])
            outputs.append(result)
        return outputs
    
    def __classify_fc(self, output):
        acc_values = self.softmax(output)
        class_id = torch.argmax(acc_values, dim=1)
        return class_id, acc_values

    def __classify_embedding(self, embedding):
        diff = 1 - F.cosine_similarity(embedding, self.data_base, dim=1)
        val, indi = torch.sort(diff, descending = False)
        
        embedding_classification_output = []

        for idx in range(self.top_k):
            indiece = indi[idx]
            acc = round(self.calculate_probability(val[idx].item()), 3)
            
            internal_id   = self.internal_ids[indiece]
            image_id      = self.image_ids[indiece]
            annotation_id = self.annotation_ids[indiece]
            drawn_fish_id = self.drawn_fish_ids[indiece]
 
            class_info_map  = {
                'name': self.keys[internal_id]['label'],
                'species_id': self.keys[internal_id]['species_id'],
                'distance': acc,
                'accuracy':0.0,
                'image_id': image_id,
                'annotation_id': annotation_id,
                'drawn_fish_id': drawn_fish_id,
            }
    
            embedding_classification_output.append(class_info_map)
        return embedding_classification_output