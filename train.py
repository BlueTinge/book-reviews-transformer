import torch
from torch import optim
from torch import nn
from dataloader import get_imdb
from dataloader import get_amazon
from model import Net
import numpy as np

try:
    # try to import tqdm for progress updates
    from tqdm import tqdm
except ImportError:
    # on failure, make tqdm a noop
    def tqdm(x):
        return x

try:
    # try to import visdom for visualisation of attention weights
    import visdom
    from helpers import plot_weights
    vis = visdom.Visdom()
except ImportError:
    vis = None
    pass

def sigmoid(x):
    if x >= 0:
        z = np.exp(-x)
        return 1 / (1 + z)
    else:
        z = np.exp(x)
        return z / (1 + z)

def val(model,test,vocab,device):
    """
        Evaluates model on the test set
    """
    model.eval()

    print("\nValidating..")

    if not vis is None:
        visdom_windows = None
    with torch.no_grad():
        correct = 0.0
        correct_simple = 0.0
        log_loss = 0.0
        total = 0.0

        one_star_correct = 0.0
        one_star_total = 0.0

        two_star_correct = 0.0
        two_star_total = 0.0

        three_star_correct = 0.0
        three_star_total = 0.0

        four_star_correct = 0.0
        four_star_total = 0.0

        five_star_correct =0.0
        five_star_total = 0.0

        one_two_correct = 0.0
        four_five_correct = 0.0

        for i,b in enumerate(tqdm(test)):
            if not vis is None and i == 0:
                visdom_windows = plot_weights(model,visdom_windows,b,vocab,vis)

            model_out = model(b.text[0].to(device)).to("cpu").numpy()
            actual = model_out.argmax(axis=1)
            expected = b.label.numpy()
            for k in range(len(actual)):
                y = expected[k]
                minisum = 0.0
                y_star = model_out[k].argmax()
                log_loss += sigmoid(np.log(model_out[k].max()))
                bad_sum = model_out[k][0] + model_out[k][1]
                neutral = model_out[k][2]
                good_sum = model_out[k][3] + model_out[k][4]
                amax = 0
                if bad_sum >= neutral and bad_sum >= good_sum:
                    amax = 0
                elif neutral >= bad_sum and neutral >= good_sum:
                    amax = 1
                else:
                    amax = 2

                if (amax == 0 and y < 2) or (amax == 1 and y == 2) or (amax == 2 and y > 2):
                    correct_simple += 1

                if y == 0:
                    if y_star == y:
                        one_star_correct += 1
                    if amax == 0:
                        one_two_correct += 1
                    one_star_total += 1
                elif y == 1:
                    if y_star == y:
                        two_star_correct += 1
                    if amax == 0:
                        one_two_correct += 1
                    two_star_total += 1
                elif y == 2:
                    if y_star == y:
                        three_star_correct += 1
                    three_star_total += 1
                elif y == 3:
                    if y_star == y:
                        four_star_correct += 1
                    if amax == 2:
                        four_five_correct += 1
                    four_star_total += 1
                else:
                    if y_star == y:
                        five_star_correct += 1
                    if amax == 2:
                        four_five_correct += 1
                    five_star_total += 1

            correct += (actual == b.label.numpy()).sum()
            total += b.label.size(0)
        print("Log-Loss: {}\n".format(log_loss / total))
        print("Accuracy: {}%, {}/{}\n".format(correct / total * 100, correct, total))
        print("Simple: {}%, {}/{}\n".format(correct_simple / total * 100, correct_simple, total))
        print("One-Star: {}%, {}/{}".format(one_star_correct / one_star_total * 100, one_star_correct, one_star_total))
        print("Two-Star: {}%, {}/{}".format(two_star_correct / two_star_total * 100, two_star_correct, two_star_total))
        print("Three-Star: {}%, {}/{}".format(three_star_correct / three_star_total * 100, three_star_correct, three_star_total))
        print("Four-Star: {}%, {}/{}".format(four_star_correct / four_star_total * 100, four_star_correct, four_star_total))
        print("Five-Star: {}%, {}/{}".format(five_star_correct / five_star_total * 100, five_star_correct, five_star_total))
        print("Good: {}%, {}/{}".format(one_two_correct / (one_star_total + two_star_total) * 100, one_two_correct, (one_star_total + two_star_total)))
        print("Bad: {}%, {}/{}".format(four_five_correct / (four_star_total + five_star_total) * 100, four_five_correct, (four_star_total + five_star_total)))

def train(max_length,model_size,
            epochs,learning_rate,
            device,num_heads,num_blocks,
            dropout,train_word_embeddings,
            batch_size, model_path, load_model):
    """
        Trains the classifier on the IMDB sentiment dataset
    """
    train, test, vectors, vocab = get_amazon(batch_size,max_length=max_length)

    model = Net(
                model_size=model_size,embeddings=vectors,
                max_length=max_length,num_heads=num_heads,
                num_blocks=num_blocks, dropout=dropout,
                train_word_embeddings=train_word_embeddings,
                n_classes=5,
                ).to(device)

    optimizer = optim.Adam((p for p in model.parameters() if p.requires_grad),lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    if load_model:
        print("Loading model from {}...".format(load_model))
        model = torch.load(load_model)
        model.eval()
        # Validate after loading
        val(model,test,vocab,device)

    # val(model, test, vocab, device)
    #
    # for i in range(0,epochs+1):
    #     loss_sum = 0.0
    #     model.train()
    #     print("\n\nTraining epoch {}...".format(i))
    #     for j,b in enumerate(iter(tqdm(train))):
    #         optimizer.zero_grad()
    #         model_out = model(b.text[0].to(device))
    #         loss = criterion(model_out,b.label.to(device))
    #         loss.backward()
    #         optimizer.step()
    #         loss_sum += loss.item()
    #     print("Epoch: {}, Loss mean: {}\n".format(i,j,loss_sum / (j+1)))
    #
    #     # Validate on test-set every epoch
    #     val(model,test,vocab,device)
    #
    #     #Save model
    #     if model_path:
    #         print("Saving model to {}...".format(model_path))
    #         torch.save(model, model_path)

        # Load model and test
        #model = torch.load(model_path)
        #model.eval()
        # Validate again to see if its the same
        #val(model,test,vocab,device)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Train a Transformer network for sentiment analysis")
    
    ap.add_argument("--max_length",default=500,type=int,help="Maximum sequence length, \
                                                                    sequences longer than this are truncated")
    
    ap.add_argument("--model_size",default=128,type=int,help="Hidden size for all \
                                                                    hidden layers of the model")
    
    ap.add_argument("--epochs",default=1000,type=int,help="Number of epochs to train for")

    ap.add_argument("--learning_rate",default=0.001,type=float,dest="learning_rate",help="Learning rate for optimizer")
    
    ap.add_argument("--device",default="cuda:0",dest="device",help="Device to use for training \
                                                                    and evaluation e.g. (cpu, cuda:0)")
    ap.add_argument("--num_heads",default=4,type=int,dest="num_heads",help="Number of attention heads in \
                                                                    the Transformer network")
    
    ap.add_argument("--num_blocks",default=1,type=int,dest="num_blocks",help="Number of blocks in the Transformer network")
    
    ap.add_argument("--dropout",default=0.1,type=float,dest="dropout",help="Dropout (not keep_prob, but probability of ZEROING \
                                                                    during training, i.e. keep_prob = 1 - dropout)")

    ap.add_argument("--train_word_embeddings",type=bool,default=True,dest="train_word_embeddings",help="Train word embeddings during training")
    
    ap.add_argument("--batch_size",type=int,default=128,help="Batch size")

    ap.add_argument("--model_path",default="saved.model",help="Path to save model to after every epoch. Give empty string to disable saving.")

    ap.add_argument("--load_model", default="", dest="load_model", help="If given, will load the model from this path")

    args = vars(ap.parse_args())

    train(**args)



