import torch
from tqdm import tqdm

class trainTest():

    def __init__(self, model, device, criterion, optimizer, banckmark_name, network, fold, save = False):
        self.model = model
        self.device = device
        self.criterion = criterion
        self.optimizer = optimizer
        self.path = f"./tests/{banckmark_name}/{network}_{fold}"
        self.network = network
        self.save = save


    def check_accuracy(self, loader, model):

        num_correct = 0
        num_samples = 0
        model.eval()

        with torch.no_grad():
            for x, y in loader:
                original_y = y
                x = x.to(device=self.device)
                y = y.to(device=self.device)

                scores = model(x)
                probs = torch.nn.functional.softmax(scores, dim=1)
                conf, predictions = torch.max(probs, 1)
                #predictions = torch.tensor([1 if i[1] >= i[0] else 0 for i in scores]).to(device)
                num_correct += (predictions == y).sum()
                num_samples += predictions.size(0)
        accuracy = float(num_correct)/float(num_samples)*100
        return accuracy, conf, predictions, original_y, probs



    def check_full_accuracy(self, loader):

        num_correct = 0
        num_samples = 0
        self.model.eval()
        all_conf = []
        all_pred = []
        all_original_y = []
        all_probs = []

        with torch.no_grad():
            for x, y in loader:
                original_y = y
                x = x.to(device=self.device)
                y = y.to(device=self.device)

                scores = self.model(x)
                probs = torch.nn.functional.softmax(scores, dim=1)
                conf, predictions = torch.max(probs, 1)
                num_correct += (predictions == y).sum()
                num_samples += predictions.size(0)
                all_conf.extend(conf)
                all_pred.extend(predictions)
                all_original_y.extend(y)
                all_probs.extend(probs)
        accuracy = float(num_correct)/float(num_samples)*100
        return accuracy, all_conf, all_pred, all_original_y, all_probs

    def train(self, train_loader, validation_loader, num_epochs):
        self.model.train()
        bestAcc = 0.0
        for epoch in range(num_epochs):
            #print(f"Epoch {epoch}")
            loop = tqdm(train_loader, total = len(train_loader), leave = True)
            if epoch % 2 == 0:
                val_acc, conf, predictions, yGT, probs = self.check_accuracy(validation_loader, self.model)
                bestAcc = val_acc if val_acc>bestAcc else bestAcc
                self.model.train()
            for imgs, labels in loop:
                imgs = imgs.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(imgs)
                loss = self.criterion(outputs, labels)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                loop.set_description(f"Epoch [{epoch}/{num_epochs}]")
                loop.set_postfix(loss = loss.item())

        #torch.save(self.model, self.path)
        if self.save:
            model_scripted = torch.jit.script(self.model)  # Export to TorchScript
            model_scripted.save(f'{self.path}.pt')  # Save
        #val_acc, conf, predictions, yGT, probs = self.check_full_accuracy(validation_loader, self.model)
        #self.model.train()
        #return bestAcc#, conf, predictions, yGT, probs

    def reset_weights(self, m):
      '''
        Try resetting model weights to avoid
        weight leakage.
      '''
      for layer in m.children():
       if hasattr(layer, 'reset_parameters'):
        #print(f'Reset trainable parameters of layer = {layer}')
        layer.reset_parameters()
