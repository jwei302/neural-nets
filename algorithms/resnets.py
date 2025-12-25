import torch 
from utils import plot_images

class ResNetBlock(torch.nn.Module):
    '''
    Basic ResNet block class

    Arg(s):
        in_channels : int
            number of input channels
        out_channels : int
            number of output channels
        stride : int
            stride of convolution
    '''

    def __init__(self,
                 in_feature,
                 out_channels,
                 stride=1):
        super(ResNetBlock, self).__init__()

        # Deep Residual Learning for Image Recognition: https://arxiv.org/pdf/1512.03385.pdf

        self.conv1 = torch.nn.Conv2d(in_feature, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(out_channels)
        self.relu = torch.nn.ReLU(inplace=True)

        self.conv2 = torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = torch.nn.BatchNorm2d(out_channels)

        self.conv3 = None

        if stride != 1 or in_feature != out_channels:
            self.projection = torch.nn.Sequential(
                torch.nn.Conv2d(in_feature, out_channels, kernel_size=1, stride=stride, bias=False),
                torch.nn.BatchNorm2d(out_channels)
            )
        else:
            self.projection = torch.nn.Sequential()

    def forward(self, x):
        '''
        Forward input x through a basic ResNet block

        Arg(s):
            x : torch.Tensor[float32]
                N x C x H x W input tensor
        Returns:
            torch.Tensor[float32] : N x K x h x w output tensor
        '''

        identity = self.projection(x)
        output = torch.relu(self.bn1(self.conv1(x)))
        output = self.bn2(self.conv2(output))
        output = torch.relu(output + identity)
        return output


class ResNet18(torch.nn.Module):
    '''
    ResNet18 convolutional neural network

    Arg(s):
        n_input_channel : int
            number of channels in input data
        n_filters : list
            number of filters to use for each block
    '''

    def __init__(self, n_input_feature, n_output):
        super(ResNet18, self).__init__()

        # Based on https://arxiv.org/pdf/1512.03385.pdf
        self.conv = torch.nn.Conv2d(n_input_feature, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn = torch.nn.BatchNorm2d(64)
        self.relu = torch.nn.ReLU(inplace=True)

        self.maxpool = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(in_feature=64, out_channels=64, num_blocks=2, stride=1)
        self.layer2 = self._make_layer(in_feature=64, out_channels=128, num_blocks=2, stride=2)
        self.layer3 = self._make_layer(in_feature=128, out_channels=256, num_blocks=2, stride=2)
        self.layer4 = self._make_layer(in_feature=256, out_channels=512, num_blocks=2, stride=2)

        self.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.linear = torch.nn.Linear(512, n_output)

    def _make_layer(self, in_feature, out_channels, num_blocks, stride):
        # first block does downsampling, the other blocks refine
        strides = [stride] + [1] * (num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(ResNetBlock(in_feature, out_channels, stride))
            in_feature = out_channels
        return torch.nn.Sequential(*layers)

    def forward(self, x):
        '''
        Forward input x through a ResNet encoder

        Arg(s):
            x : torch.Tensor[float32]
                N x C x H x W input tensor
        Returns:
            torch.Tensor[float32] : N x K x h x w output tensor
        '''

        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, start_dim=1)
        x = self.linear(x)

        return x
    
def train(net,
          dataloader,
          n_epoch,
          optimizer,
          learning_rate_decay,
          learning_rate_decay_period,
          device):
    '''
    Trains the network using a learning rate scheduler

    Arg(s):
        net : torch.nn.Module
            neural network or ResNet
        dataloader : torch.utils.data.DataLoader
            # https://pytorch.org/docs/stable/data.html
            dataloader for training data
        n_epoch : int
            number of epochs to train
        optimizer : torch.optim
            https://pytorch.org/docs/stable/optim.html
            optimizer to use for updating weights
        learning_rate_decay : float
            rate of learning rate decay
        learning_rate_decay_period : int
            period to reduce learning rate based on decay e.g. every 2 epoch
        device : str
            device to run on
    Returns:
        torch.nn.Module : trained network
    '''

    device = 'cuda' if device == 'gpu' or device == 'cuda' else 'cpu'
    device = torch.device(device)

    net.to(device)

    # https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
    loss_func = torch.nn.CrossEntropyLoss()

    for epoch in range(n_epoch):
        total_loss = 0.0

        if epoch and epoch % learning_rate_decay_period == 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= learning_rate_decay

        for batch, (images, labels) in enumerate(dataloader):
            images = images.to(device)
            labels = labels.to(device)

            outputs = net.forward(images)

            optimizer.zero_grad()

            loss = loss_func(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        mean_loss = total_loss / len(dataloader)

        # Log average loss over the epoch
        print('Epoch={}/{}  Loss: {:.3f}'.format(epoch + 1, n_epoch, mean_loss))

    return net

def evaluate(net, dataloader, class_names, device):
    '''
    Evaluates the network on a dataset

    Arg(s):
        net : torch.nn.Module
            neural network
        dataloader : torch.utils.data.DataLoader
            # https://pytorch.org/docs/stable/data.html
            dataloader for training data
        class_names : list[str]
            list of class names to be used in plot
        device : str
            device to run on
    '''

    device = 'cuda' if device == 'gpu' or device == 'cuda' else 'cpu'
    device = torch.device(device)

    net.to(device)

    n_correct = 0
    n_sample = 0

    # Make sure we do not backpropagate
    with torch.no_grad():

        for (images, labels) in dataloader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = net(images)
            predicted_labels = torch.argmax(outputs, dim=1)

            # Accumulate number of samples
            n_sample = n_sample + labels.shape[0]

            n_correct = (predicted_labels == labels).sum().item() + n_correct

    mean_accuracy = n_correct / n_sample

    print('Mean accuracy over {} images: {:.3f}%'.format(n_sample, mean_accuracy*100))

    images = images.cpu().numpy()

    images = images.transpose(0, 2, 3, 1)

    # map them to their corresponding class labels
    labels = labels.cpu().numpy()
    labels = [class_names[label] for label in labels]

    # map them to their corresponding class labels
    outputs = outputs.cpu().numpy()
    outputs = np.argmax(outputs, axis=1)
    outputs = [class_names[output] for output in outputs]


    # Convert images, outputs and labels to a lists of lists
    grid_size = 5

    images_display = []
    subplot_titles = []

    for row_idx in range(grid_size):
        idx_start = row_idx * grid_size
        idx_end = idx_start + grid_size

        images_display.append(images[idx_start:idx_end])
        titles = [f'output={outputs[i]}\nlabel={labels[i]}' for i in range(idx_start, idx_end)]

        subplot_titles.append(titles)

    plot_images(images_display, grid_size, grid_size, subplot_titles)
