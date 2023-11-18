

# Standard library imports
from os import makedirs, rmdir
from os.path import exists as pathexists, join as pathjoin
from random import shuffle

# External library imports
import numpy as np
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter

# torchvision related imports
from torchvision.datasets import CelebA
from torchvision import transforms

# Custom module import
# module containing pre-defined neural network models
from res_facenet.models import model_921 as facenet

# itertools import
from itertools import chain


"""
Utility class for tracking and updating a running score based on scoring function. 
"""
class Metric:
    # Constructor method
    def __init__(self, scoring_fun=lambda x: x):
        # Initialize running score to 0
        self.running_score = 0
        # Set the scoring function, defaulting to identity function if not provided
        self.scoring_fun = scoring_fun

    # Method to update the running score
    def update(self, *args):
        # Apply the scoring function to the input arguments
        score = self.scoring_fun(*args)
        """
        # Update the running score using a weighted sum
        The running score is updated using a weighted sum. 
        It is a combination of the previous running score (70% weight) and the new score (30% weight). 
        This gives more importance to the previous score, making the metric somewhat resistant to abrupt changes.
        """
        self.running_score = self.running_score * 0.7 + score * 0.3

    # Method to get the current running score
    def output(self):
        # Return the current running score
        return self.running_score


class AttrHider():
    
    """
    This method defines an encoder neural network using the PyTorch nn.Sequential container. 
    The encoder takes an input with 3 channels (RGB image) and produces a 1-dimensional output. 
    """
    @staticmethod
    def Encoder():

        """
        Define a sequential neural network model for encoding
        """
        return nn.Sequential(
            # Convolutional layer: 3 input channels, 64 output channels, kernel size 4x4, stride 2, padding 1
            nn.Conv2d(3, 64, 4, 2, padding=1),
            nn.InstanceNorm2d(64),  # Instance normalization for the first layer
            nn.LeakyReLU(0.3),  # Leaky ReLU activation with a negative slope of 0.3

            # Similar structure for the next layers with increasing output channels
            nn.Conv2d(64, 128, 4, 2, padding=1),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.3),

            nn.Conv2d(128, 256, 4, 2, padding=1),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.3),

            nn.Conv2d(256, 512, 4, 2, padding=1),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.3),

            nn.Conv2d(512, 1024, 4, 2, padding=1),
            nn.InstanceNorm2d(1024),
            nn.LeakyReLU(0.3),

            nn.MaxPool2d(7),  # Max pooling layer with kernel size 7x7, reducing the spatial dimensions.
            nn.Flatten(),  # Flatten the output tensor, to prepare it for the fully connected layer.
            nn.Linear(1024, 1)  # Fully connected layer with 1024 input features and 1 output feature.
            # This part of the network is intended for binary classification ?
            # (e.g., age attribute present or not).
        )

    

    class Generator(nn.Module):
        def __init__(self):
            super().__init__()

            # Initialize lists to hold encoder and decoder layers
            self.encoder_layers = nn.ModuleList()
            self.decoder_layers = nn.ModuleList()

            # Define the number of channels for each layer
            channels = [3, 64, 128, 256, 512, 1024]

            # Create encoder and decoder layers using a loop
            for i in range(5):
                # Encoder layers: Convolution, Instance Normalization, Leaky ReLU
                self.encoder_layers.append(nn.Sequential(
                        nn.Conv2d(channels[i], channels[i+1], 4, 2, padding=1),
                        nn.InstanceNorm2d(channels[i+1]),
                        nn.LeakyReLU()
                    ))

                # Decoder layers: Transposed Convolution, Instance Normalization, Leaky ReLU
                self.decoder_layers.append(nn.Sequential(
                        nn.ConvTranspose2d(channels[-i-1], channels[-i-2], 4, 2, padding=1),
                        nn.InstanceNorm2d(channels[-i-2]),
                        nn.LeakyReLU()
                    ))

            # Normalization layer
            self.norm_layer = nn.Sequential(
                    nn.Flatten(),
                    nn.BatchNorm1d(1024*7*7, affine=False),
                    nn.Unflatten(1, (1024, 7, 7))
                )

        def forward(self, x):
            encoder_outputs = []  # For skip connections

            # Forward pass through the encoder layers
            for l in self.encoder_layers:
                x = l(x)
                encoder_outputs.append(x)

            # Set x between 0 and 1
            x = self.norm_layer(x)
            x = torch.abs(torch.rand(len(x), 1024, 7, 7, device='cuda') - x)

            # Forward pass through the decoder layers with skip connections
            for l in self.decoder_layers:
                x = torch.cat([x, encoder_outputs.pop()], 1)  # Concatenate with the encoder output
                x = l[0](x)  # Apply the first layer in the decoder

            # Return the final output tensor.
            return x

    

    '''
    This constructor sets up the entire training pipeline, including network architectures, 
    hyperparameters, dataset loading, and metrics for monitoring training progress.

    Parameters
    ----------
    savedir : str, optional
        Directory that checkpoint files and tensorboard info is saved to. The default is 'Output'.

    attr_id : int, optional
        Attribute that is being loaded. The default is -1.

    Returns
    -------
    AttrHider object (for training attribute hider)
    '''
    def __init__(self, savedir = 'Output', attr_id = -1):

        # Build networks, Network Initialization:
        # Initialization of several neural networks, including classifier, discriminator, generator, and identifier.
        # The corresponding optimizers (optimizer_D and optimizer_G) are also initialized.
        self.classifier = self.Encoder()
        self.classifier = self.classifier.cuda()
        self.discriminator = self.Encoder()
        self.discriminator = self.discriminator.cuda() #1 if real, 0 if generated
        self.generator = self.Generator().cuda()
        self.identifier = facenet().requires_grad_(False).cuda()
        self.optimizer_D = torch.optim.Adam(chain(self.discriminator.parameters(),self.classifier.parameters()),lr = 0.002, betas=(0.5,0.99))
        self.optimizer_G = torch.optim.Adam(self.generator.parameters(), lr = 0.02,betas=(0.5,0.99))
        
        # Training hyperparameters
        # Various hyperparameters related to training are initialized, 
        # such as coefficients for different components of the loss.
        self.gradient_coeff = 0
        self.classifier_coeff = 1
        self.protected_image_classifier_coeff = 0
        self.discriminator_coeff = 0
        self.generator_vs_discriminator_coeff = 0
        self.generator_vs_classifier_coeff = 0
        self.id_coeff = 0
        
        # Get dataset
        # CelebA dataset is loaded using specified transformations, and a dataloader is created
        transform = transforms.Compose((transforms.ToTensor(),
            transforms.CenterCrop([170,170]),
            transforms.Resize(size=(224, 224), antialias=True), 
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ))
        dataset = CelebA('/home/maya/Desktop/datasets/',transform=transform)
        
        self.dataloader = torch.utils.data.DataLoader(dataset, batch_size = 32, num_workers = 2, shuffle=True)

        # Build Metrics
        # Metrics for tracking generator and discriminator losses, accuracy, and hidden accuracy are initialized.
        self.G_loss = Metric()
        self.D_loss = Metric()
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.identity_loss_fn = nn.CosineEmbeddingLoss()
        def acc(y_true, y_pred): 
            return np.mean(y_true==(y_pred>0))  
        self.acc_metric = Metric(acc)
        self.hidden_acc_metric = Metric(acc)
        
        #Set up / load from save directory.
        self.savedir = savedir
        
        # Set up tensorboard
        # TensorBoard is set up with a summary writer to save training logs.
        self._step = 0 
        self.writer = SummaryWriter(savedir)
        
        # Other variables: attribute ID is set.
        self._attr_id = attr_id
    

    def get_config(self):
        return {
                'attr_id' : self._attr_id                
            }
    

    """
    This private method zeros the gradients for the classifier, discriminator, and generator.
    """
    def _zero_grad(self):
        # Zero gradients for classifier, discriminator, and generator
        self.classifier.zero_grad()
        self.discriminator.zero_grad()
        self.generator.zero_grad()


    """
    The method performs a training step for the model. It iterates over batches in the dataloader.
    """
    def train_step(self, n=6):
        def calc_gradient(network, orig_images, protected_images):
            # Calculate gradient for the network using a random epsilon
            epsilon = torch.rand(len(orig_images), 1, 1, 1, device='cuda').expand(-1, 3, 224, 224)
            merged_images = epsilon * orig_images + (1 - epsilon) * protected_images
            network(merged_images).backward(gradient=torch.ones([len(merged_images), 1], device='cuda'), inputs=merged_images)
            return merged_images.grad

        for i, (x, y) in enumerate(self.dataloader):
            # Gradients are zeroed.
            self._zero_grad()  

            # Original and protected images are obtained from the generator.
            orig_images = x.to('cuda')
            classifications = y.type(torch.float)[:, self._attr_id].to('cuda')
            if i >= n:
                break
            protected_images = self.generator(orig_images)

            # Discriminator loss is calculated, including a gradient penalty for regularization.
            loss = 0
            grad_x = calc_gradient(self.discriminator, orig_images, protected_images)
            gradient_penalty = self.gradient_coeff * torch.square(
                torch.norm(grad_x, 2, dim=1) - 1).mean()  # Gradient penalty for regularization

            disc_loss = self.discriminator(protected_images) - self.discriminator(orig_images)
            disc_loss += gradient_penalty
            disc_loss = disc_loss.mean()
            loss += self.discriminator_coeff * disc_loss

            ind_has_attr = classifications == 1
            num_with_attr = int(classifications.sum().detach())

            if num_with_attr > 0 and num_with_attr < 32:
                # Classifier loss is calculated based on the protected images and original images, 
                # considering attribute presence.
                orig_images_attr = self.classifier(orig_images).flatten()
                protected_images_attr = self.classifier(protected_images).flatten()

                attr_loss = self.protected_image_classifier_coeff * 0.4527663019067221 * self.loss_fn(
                    protected_images_attr[ind_has_attr], torch.ones((num_with_attr), device='cuda'))
                attr_loss += 0.4527663019067221 * self.loss_fn(orig_images_attr[ind_has_attr],
                                                            torch.ones((num_with_attr), device='cuda'))
                attr_loss += self.protected_image_classifier_coeff * 0.5472336980932778 * self.loss_fn(
                    protected_images_attr[~ind_has_attr],
                    torch.zeros((len(ind_has_attr) - num_with_attr), device='cuda'))
                attr_loss += 0.5472336980932778 * self.loss_fn(orig_images_attr[~ind_has_attr],
                                                                torch.zeros((len(ind_has_attr) - num_with_attr),
                                                                            device='cuda'))
                loss += self.classifier_coeff * attr_loss
            else:
                continue
            
            # Gradients are computed and used for optimization.
            loss.backward()
            self.optimizer_D.step()
            self.D_loss.update(loss.detach().cpu().numpy())

            # Metrics and tensorboard logs are updated.
            self.writer.add_scalar('loss_discriminator', disc_loss.detach().cpu().numpy(), self._step)
            self.writer.add_scalar('loss_classifier', attr_loss.detach().cpu().numpy(), self._step)
            self._step += 1

            if self._step % 10 == 0:  # update every 10 steps
                self.writer.add_histogram('classifier_output', orig_images_attr, self._step)
                self.writer.add_histogram('current_classifier_output', orig_images_attr)
            if self._step % 100 == 0:  # update every 100 steps
                self.writer.add_image('generated_image', protected_images[0], self._step)
                self.writer.add_image('original_image', orig_images[0], self._step)
                self._zero_grad()


        # The generator is trained based on the confusion strategy with the discriminator and classifier.
        protected_images = self.generator(orig_images)
        protected_images_id_features = self.identifier(protected_images)
        orig_images_id_features = self.identifier(orig_images)
        protected_image_classifications = self.classifier(protected_images).flatten()

        # Relevant losses are calculated, and the generator is optimized.
        G_loss = -self.discriminator(protected_images).mean()
        attr_loss = -self.loss_fn(protected_image_classifications, classifications)
        id_loss = self.identity_loss_fn(protected_images_id_features, orig_images_id_features,
                                        torch.ones(1).cuda())
        loss = self.generator_vs_discriminator_coeff * G_loss + self.generator_vs_classifier_coeff * attr_loss + self.id_coeff * id_loss
        loss.backward()
        self.optimizer_G.step()
        self.G_loss.update(loss.detach().cpu().numpy())
        self.acc_metric.update(classifications.cpu().numpy(),
                            self.classifier(orig_images).flatten().detach().cpu().numpy())
        self.hidden_acc_metric.update(classifications.cpu().numpy(),
                                    protected_image_classifications.detach().cpu().numpy())

        # Metrics and tensorboard logs are updated.
        self.writer.add_scalar('loss_confuse_discriminator', G_loss.detach().cpu().numpy(), self._step)
        self.writer.add_scalar('loss_confuse_classifier', attr_loss.detach().cpu().numpy(), self._step)
        self.writer.add_scalar('loss_id', id_loss.detach().cpu().numpy(), self._step)

    
    '''
        Parameters
        ----------
        steps : int, optional
            Number of iterations that one trains. The default is 100000.
        verbose : bool, optional
            Whether to print out parameters. The default is True.
        save_chkpt_freq : TYPE, optional
            How often to save checkpoint files. The default value (0) represents no saving checkpoint files.
        **kwargs :
            Parameters to feed to self.train_step

        Returns
        -------
        None.
    '''
    def train(self, steps=100000, verbose=True, save_chkpt_freq=0, **kwargs):

        # Display header if verbose is True
        if verbose:
            print('i Discriminator Loss   Generator Loss  AccOrig AccProtected')

        # Training loop: iterates over the specified number of steps 
        for i in range(steps):
            # calls the train_step method to perform a single training step
            self.train_step(**kwargs) 

            # Save checkpoint if save_chkpt_freq is set
            if save_chkpt_freq:
                if i % save_chkpt_freq == 0:
                    self.save(str(self._step) + '.pt')  # Save checkpoint file

            # Display progress every 100 steps if verbose is True
            if i % 100 == 0:
                if verbose:
                    # Print iteration, discriminator loss, generator loss, accuracy on original images, and accuracy on protected images
                    print(i, self.D_loss.output(), self.G_loss.output(), self.acc_metric.output(),
                        self.hidden_acc_metric.output())

    

    '''
        Change the coefficients used during training.
        The coefficients control the impact of different components on the overall loss during training. 
        This method allows for easy adjustment of these coefficients during runtime.

        Parameters
        ----------
        coeffs : list
            List of coefficients to set for different components.

        Returns
        -------
        None.
    '''
    def change_coeff(self, coeffs):
        # Set the coefficients for different components
        self.classifier_coeff = coeffs[0]
        self.protected_image_classifier_coeff = coeffs[1]
        self.discriminator_coeff = coeffs[2]
        self.gradient_coeff = coeffs[3]
        self.generator_vs_discriminator_coeff = coeffs[4]
        self.generator_vs_classifier_coeff = coeffs[5]
        self.id_coeff = coeffs[6]

    
    '''
        Save the current state of the GAN model.
        Including the state dictionaries of the classifier, discriminator, generator, and their respective optimizers.

        Parameters
        ----------
        filename : str or None, optional
            Name of the file to save the model. If None, the default filename is 'GAN.pt'.
            The default is None.

        Returns
        -------
        None.
    '''
    def save(self, filename=None):
        # Set default filename if not provided
        if filename is None:
            filename = 'GAN.pt'

        # Save the state of various model components
        torch.save([
            self.classifier.state_dict(),
            self.discriminator.state_dict(),
            self.generator.state_dict(),
            self.optimizer_D.state_dict(),
            self.optimizer_G.state_dict()
        ], os.path.join(self.savedir, filename))


    '''
        Load the state of the GAN model from a saved file.
        The saved file is assumed to contain a list of state dictionaries in the order: 
        [classifier, discriminator, generator, optimizer_D, optimizer_G].

        Parameters
        ----------
        filename : str or None, optional
            Name of the file from which to load the model state. If None, the default filename is 'GAN.pt'.
            The default is None.

        Returns
        -------
        None.
    '''
    def load(self, filename=None):
        # Set default filename if not provided
        if filename is None:
            filename = 'GAN.pt'

        # Load the saved state from the specified file
        params = torch.load(filename)

        # Update the state of various model components
        self.optimizer_G.load_state_dict(params.pop())
        self.optimizer_D.load_state_dict(params.pop())
        self.generator.load_state_dict(params.pop())
        self.discriminator.load_state_dict(params.pop())
        self.classifier.load_state_dict(params.pop())



if __name__ == '__main__':
    # Create an instance of the AttrHider class
    m = AttrHider()

    # Train classifier
    # Set the coefficients for training the classifier ([1, 0, 0, 0, 0, 0, 0])
    # This means only the classifier will be trained, with no influence from other components
    m.change_coeff([1, 0, 0, 0, 0, 0, 0])

    # Train the classifier for 5000 iterations
    m.train(5000)