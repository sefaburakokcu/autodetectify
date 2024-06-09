import torch
from abc import ABC, abstractmethod
from transformers import AutoProcessor, OwlViTForObjectDetection
from transformers import Owlv2Processor, Owlv2ForObjectDetection
from transformers import GroundingDinoProcessor, GroundingDinoForObjectDetection


class ZeroShotObjectDetectionModel(ABC):
    def __init__(self, model_name: str, device: str = 'cuda'):
        self.model_name = model_name
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = self.load_model()
        self.processor = self.load_processor()

    @abstractmethod
    def load_model(self):
        pass

    @abstractmethod
    def load_processor(self):
        pass

    @abstractmethod
    def predict(self, images, queries, query_type='text'):
        pass

    def preprocess(self, images, queries, query_type='text'):
        if query_type == 'text':
            inputs = self.processor(text=queries, images=images, return_tensors="pt").to(self.device)
        elif query_type == 'image':
            inputs = self.processor(query_images=queries, images=images, return_tensors="pt").to(self.device)
        else:
            raise ValueError("query_type must be either 'text' or 'image'")
        return inputs

    def postprocess(self, outputs, target_sizes, query_type='text', threshold=0.1, nms_threshold=0.3):
        if query_type == 'text':
            results = self.processor.post_process_object_detection(outputs=outputs,
                                                                   threshold=threshold,
                                                                   target_sizes=target_sizes)
        elif query_type == 'image':
            results = self.processor.post_process_image_guided_detection(outputs=outputs,
                                                                         threshold=threshold,
                                                                         nms_threshold=nms_threshold,
                                                                         target_sizes=target_sizes)
        else:
            raise ValueError("query_type must be either 'text' or 'image'")

        return results


class OwlViTZeroShotObjectDetectionModel(ZeroShotObjectDetectionModel):
    def __init__(self, model_name: str = "google/owlvit-base-patch32", device: str = 'cuda', threshold: float = 0.1):
        super().__init__(model_name, device)
        self.threshold = threshold

    def load_model(self):
        model = OwlViTForObjectDetection.from_pretrained(self.model_name)
        model.to(self.device)
        return model

    def load_processor(self):
        processor = AutoProcessor.from_pretrained(self.model_name)
        return processor

    def predict(self, images, queries, query_type='text'):
        inputs = self.preprocess(images, queries, query_type)
        with torch.no_grad():
            if query_type == 'text':
                outputs = self.model(**inputs)
            elif query_type == 'image':
                outputs = self.model.image_guided_detection(**inputs)
            else:
                raise ValueError("query_type must be either 'text' or 'image'")

        target_sizes = torch.Tensor([image.size[::-1] for image in images]).to(self.device)
        return self.postprocess(outputs, target_sizes, query_type, self.threshold)


class GroundingDINOZeroShotObjectDetectionModel(ZeroShotObjectDetectionModel):
    def __init__(self, model_name: str = "IDEA-Research/grounding-dino-base", device: str = 'cuda',
                 box_threshold: float = 0.1, text_threshold: float = 0.1):
        super().__init__(model_name, device)
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold

    def load_model(self):
        # Replace with actual model loading logic
        model = GroundingDinoForObjectDetection.from_pretrained(self.model_name)
        model.to(self.device)
        return model

    def load_processor(self):
        # Replace with actual processor loading logic
        processor = GroundingDinoProcessor.from_pretrained(self.model_name)
        return processor

    def postprocess(self, outputs, input_ids, target_sizes, query_type='text', threshold=0.1, nms_threshold=0.3,
                    box_threshold=0.25, text_threshold=0.25):
        if query_type == 'text':
             results = self.processor.post_process_grounded_object_detection(outputs=outputs,
                                                                            input_ids=input_ids,
                                                                            box_threshold=box_threshold,
                                                                            text_threshold=text_threshold,
                                                                            target_sizes=target_sizes)
        elif query_type == 'image':
            results = self.processor.post_process_image_guided_detection(outputs=outputs,
                                                                         threshold=threshold,
                                                                         nms_threshold=nms_threshold,
                                                                         target_sizes=target_sizes)
        else:
            raise ValueError("query_type must be either 'text' or 'image'")

        return results

    def predict(self, images, queries, query_type='text'):
        inputs = self.preprocess(images, queries, query_type)
        with torch.no_grad():
            outputs = self.model(**inputs)
        target_sizes = torch.Tensor([image.size[::-1] for image in images]).to(self.device)
        return self.postprocess(outputs, inputs.input_ids, target_sizes, query_type,
                                self.box_threshold, self.text_threshold)


class OwlV2ZeroShotObjectDetectionModel(ZeroShotObjectDetectionModel):
    def __init__(self, model_name: str = "google/owlv2-base-patch16", device: str = 'cuda', threshold: float = 0.1):
        super().__init__(model_name, device)
        self.threshold = threshold

    def load_model(self):
        # Replace with actual model loading logic
        model = Owlv2ForObjectDetection.from_pretrained(self.model_name)
        model.to(self.device)
        return model

    def load_processor(self):
        # Replace with actual processor loading logic
        processor = Owlv2Processor.from_pretrained(self.model_name)
        return processor

    def predict(self, images, queries, query_type='text'):
        inputs = self.preprocess(images, queries, query_type)
        with torch.no_grad():
            if query_type == 'text':
                outputs = self.model(**inputs)
            elif query_type == 'image':
                outputs = self.model.image_guided_detection(**inputs)
            else:
                raise ValueError("query_type must be either 'text' or 'image'")

        target_sizes = torch.Tensor([image.size[::-1] for image in images]).to(self.device)
        return self.postprocess(outputs, target_sizes, query_type, self.threshold)


# Example usage
if __name__ == "__main__":
    # OWL-ViT Example
    owlvit_model_name = "google/owlvit-base-patch32"
    owlvit_model = OwlViTZeroShotObjectDetectionModel(model_name=owlvit_model_name)

    # Grounding DINO Example
    grounding_dino_model_name = "IDEA-Research/grounding-dino-base"
    grounding_dino_model = GroundingDINOZeroShotObjectDetectionModel(model_name=grounding_dino_model_name)

    # OWL-ViT v2 Example
    owlv2_model_name = "google/owlv2-base-patch16"
    owlv2_model = OwlV2ZeroShotObjectDetectionModel(model_name=owlv2_model_name)

    from PIL import Image

    # Dummy data for example purposes
    images = [Image.open("../../../inputs/demo_images/DJI_0013_450.jpg")]  # list of PIL images

    # Text-guided detection
    text_queries = ["a photo of a cat", "a photo of a dog", "a photo of a cow"]

    # OWL-ViT predictions
    # owlvit_predictions = owlvit_model.predict(images, text_queries, query_type='text')
    # print("OWL-ViT text-guided predictions:", owlvit_predictions)

    # Image-guided detection
    reference_images = [Image.open("../../../inputs/demo_images/reference_cow.png")]  # list of PIL reference images
    #
    # # OWL-ViT image-guided predictions
    # owlvit_image_predictions = owlvit_model.predict(images, reference_images, query_type='image')
    # print("OWL-ViT image-guided predictions:", owlvit_image_predictions)
    # del owlvit_model
    # del owlvit_predictions

    # Grounding DINO predictions
    grounding_dino_predictions = grounding_dino_model.predict(images, text_queries[2], query_type='text')
    print("Grounding DINO text-guided predictions:", grounding_dino_predictions)
    del grounding_dino_model
    del grounding_dino_predictions

    # OWL-ViT v2 predictions
    owlv2_predictions = owlv2_model.predict(images, text_queries, query_type='text')
    print("OWL-ViT v2 text-guided predictions:", owlv2_predictions)

    # # OWL-ViT v2 image-guided predictions
    # owlv2_image_predictions = owlv2_model.predict(images, reference_images, query_type='image')
    # print("OWL-ViT v2 image-guided predictions:", owlv2_image_predictions)
