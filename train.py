# Imports here
import numpy as np
import torchvision as tv
import torch
import torchvision.transforms as transforms
import torch.optim as optim
import matplotlib as plt
import torch.nn as nn
from collections import OrderedDict
from PIL import Image

data_dir = 'flowers'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

# TODO: Define your transforms for the training, validation, and testing sets
mean=[0.485, 0.456, 0.406]
std=[0.229, 0.224, 0.225]
train_transform = transforms.Compose([
    transforms.RandomRotation(30),
    transforms.Resize(255),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean,std=std)])
valid_transform = transforms.Compose([
    transforms.Resize(255),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean,std=std)])
test_transform = transforms.Compose([
    transforms.Resize(255),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean,std=std)])
# TODO: Load the datasets with ImageFolder
train_set = tv.datasets.ImageFolder(train_dir, transform=train_transform)
valid_set = tv.datasets.ImageFolder(valid_dir, transform=valid_transform)
test_set = tv.datasets.ImageFolder(test_dir, transform=test_transform)
    
# TODO: Using the image datasets and the trainforms, define the dataloaders
train_loader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)
valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=True)

import json

with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)

#print(len(cat_to_name))

# TODO: Build and train your network
#alexnet = tv.models.alexnet(pretrained=True)
# Use GPU if it's available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = tv.models.vgg11(pretrained=True)
print(model)
#Freeze parameters, turn off gradient for the model
for param in model.parameters():
    param.requires_grad = False
#Define new classifier    
classifier = nn.Sequential(nn.Linear(25088,4096),
                           nn.ReLU(),
                           nn.Dropout(p=0.5),
                           nn.Linear(4096,4096),
                           nn.ReLU(),
                           nn.Dropout(p=0.5),
                           nn.Linear(4096,len(cat_to_name)),
                           nn.LogSoftmax(dim=1))
model.classifier = classifier 
print(model)

criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)
model.to(device)

epochs = 4
for epoch in range(epochs):
    train_loss = 0
    valid_loss = 0
    accuracy = 0
    
    # Training the model
    model.train()
    counter = 0
    for inputs, labels in train_loader:
        # Move to device
        inputs, labels = inputs.to(device), labels.to(device)
        # Clear optimizers
        optimizer.zero_grad()
        # Forward pass
        output = model.forward(inputs)
        # Loss
        loss = criterion(output, labels)
        # Calculate gradients (backpropogation)
        loss.backward()
        # Adjust parameters based on gradients
        optimizer.step()
        # Add the loss to the training set's rnning loss
        train_loss += loss.item()*inputs.size(0)
        
        # Print the progress of our training
        counter += 1
        print(counter, "/", len(train_loader))
        
    # Evaluating the model
    model.eval()
    counter = 0
    # Tell torch not to calculate gradients
    with torch.no_grad():
        for inputs, labels in valid_loader:
            # Move to device
            inputs, labels = inputs.to(device), labels.to(device)
            # Forward pass
            output = model.forward(inputs)
            # Calculate Loss
            valloss = criterion(output, labels)
            # Add loss to the validation set's running loss
            valid_loss += valloss.item()*inputs.size(0)
            
            # Since our model outputs a LogSoftmax, find the real 
            # percentages by reversing the log function
            output = torch.exp(output)
            # Get the top class of the output
            top_p, top_class = output.topk(1, dim=1)
            # See how many of the classes were correct?
            equals = top_class == labels.view(*top_class.shape)
            # Calculate the mean (get the accuracy for this batch)
            # and add it to the running accuracy for this epoch
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
            
            # Print the progress of our evaluation
            counter += 1
            print(counter, "/", len(valid_loader))
    
    # Get the average loss for the entire epoch
    train_loss = train_loss/len(train_loader.dataset)
    valid_loss = valid_loss/len(valid_loader.dataset)
    # Print out the information
    print('Accuracy: ', accuracy/len(valid_loader))
    print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(epoch, train_loss, valid_loss))
    
    
    # TODO: Do validation on the test set
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
model.eval()
accuracy = 0
test_loss = 0
with torch.no_grad():    
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        output = model(images)
        test_loss += criterion(output, labels).item()
        
        output = torch.exp(output)
        top_prb, top_class = output.topk(1, dim=1)
        equals = top_class == labels.view(*top_class.shape)
        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()      
        
test_loss = test_loss/len(test_loader.dataset)
print('Accuracy: ', accuracy/len(test_loader))
print('Epoch: {} \tTest Loss: {:.6f}'.format(epoch, test_loss))
model.train();

# TODO: Save the checkpoint 

def save_checkpoint(model, path):
    model.class_to_idx = train_set.class_to_idx
    checkpoint = {'input_size': 25088,
                  'output_size': len(cat_to_name),
                  #'hidden_layers': [each.out_features for each in model.hidden_layers], error: vgg object has no such attribute
                  'state_dict': model.state_dict(),
                  'optimizer': optimizer.state_dict(),
                  'class_to_idx': model.class_to_idx,
                  'classifier': classifier,
                  'classifier_state_dict': model.classifier.state_dict()}
    torch.save(checkpoint, path)
    return checkpoint
checkpoint = save_checkpoint(model,'checkpoint.pth')
#print(checkpoint)
#print(classifier)
#print(classifier.state_dict())
#print(optimizer.state_dict())
print(optimizer)

# TODO: Write a function that loads a checkpoint and rebuilds the model
def load_checkpoint(path):
    checkpoint = torch.load(path)
    model = tv.models.vgg11(pretrained=True)
    #model.hidden_layers = nn.ModuleList([nn.Linear(input_size, hidden_layers[0])])
    classifier = checkpoint['classifier']
    
    for par in model.parameters():
        par.requires_grad = False
    
    classifier.load_state_dict(checkpoint['classifier_state_dict'])
    model.classifier = classifier    
    model.load_state_dict(checkpoint['state_dict'])
    optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)
    optimizer.load_state_dict(checkpoint['optimizer'])
    return checkpoint, model, optimizer
#checkpoint, model, optimizer = load_checkpoint('checkpoint.pth')
#print(checkpoint,model,optimizer)

def process_image(image_path):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    # TODO: Process a PIL image for use in a PyTorch model
    #Load image
    image = Image.open(image_path)
    width, height = image.size
    print("original size: "+str(width)+" "+str(height) )
    # Resize to 256
    if width <= height:
        size = (256, int(256*height/width))
        image = image.resize(size)
    else:
        size = (int(256*width/height), 256)
        image = image.resize(size)
    
    #Get new width and height
    width, height = image.size
    print("new size :"+str(width)+" "+str(height))
    
    # Crop image to 224x224
    left = (width - 224)/2
    top = (height - 224)/2
    right = (width + 224)/2
    bottom = (height + 224)/2
    image = image.crop((left, top, right, bottom))
    
    #print("new size 2: "+image.size )
    np_image = np.array(image).transpose((2, 0, 1))
    np_image = np_image/255
    print(np_image.shape)
    np_image = np.delete(image, 3, 2)
    
    # Reorder colour channel
    #np_image = np_image.transpose((2, 0, 1))
    print(np_image)
    print(np_image.shape)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    print(mean.shape)
    print(std.shape)
    np_image = (np_image - mean) / std
        
    # Change to a torch tensor
    final_image = torch.FloatTensor([np_image])

    return final_image
def imshow(image, ax=None, title=None):
    import matplotlib.pyplot as plt
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.numpy().transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax

def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    # TODO: Implement the code to predict the class from an image file
    image = process_image(image_path)    
    print(image.shape)
    #image = image.unsqueeze(0)
    image = image.to(device)
    print(image.shape)
    # Send image to get output
    output = model.forward(image)
    
    # Reverse output's log function
    output = torch.exp(output)
    
    # Get the top predicted class, and the output percentage for that class
    probs, classes = output.topk((1,topk), dim=1)
    #print(probs,classes)
    return probs, classes, image

# TODO: Display an image along with the top 5 classes
model.eval()
class_names = []
# Process Image
image_path = 'input2.png'

# Give image to model to predict output
probs, classes, image = predict(image_path, model)
print('probs: {} and classes: {}'.format(probs, classes))

print(probs,classes)
print(cat_to_name)
print(model.class_to_idx)
#probs=probs.detach.numpy()
#classes=classes.detach.numpy()
print(probs,classes)
print(classes[0])
for i in classes[0]:
    class_names.append(model.class_to_idx.item(int(classes[0,i])))
print(class_names)
for c in classes:
    class_names.append(cat_to_name[c])
print('classnames: {}'.format(class_names))
# Show the image
ax = imshow(image)
plt.barh(probs, class_names)
plt.xlabel('Probability')
plt.title('Predicted Flower Names')
plt.show()
