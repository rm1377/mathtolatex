
from model_v0 import *

def test(model_path):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'


    ds_test = Im2LatexDataset(IMAGES_PATH,TEST_PATH,FORMULA_PATH,VOCAB_PATH,MAX_LEN,transforms=[Resize([IMG_HSIZE,IMG_WSIZE]),Normalize()]) # create dataset
    test_loader = torch.utils.data.DataLoader(ds_test,batch_size=BATCH_SIZE,shuffle=False,drop_last=True,num_workers=2) # creat dataloader

    decoder_model = DecoderRNN(HiddenSize,len(ds_test.vocab), EmbedSize, AttnSize).to(device) 
    encoder_model = CNNModel(HiddenSize).to(device)



    try:
        ckpt = torch.load(model_path)
        encoder_model.load_state_dict(ckpt['encoder_model'])
        decoder_model.load_state_dict(ckpt['decoder_model'])
        print("model is loaded")
    except Exception as e:
        raise ValueError('There is no checkpoint in %s'%model_path)



    criterion = torch.nn.CrossEntropyLoss() 




    def forward_pass(images, labels):
        features = encoder_model(images)
        features = features.view([BATCH_SIZE,-1,HiddenSize])

        #states = decoder_model.initHidden()
        mean_features = torch.mean(features,1)
        states = [mean_features,mean_features]
        total_loss = 0
        predictions = []
        for t in range(1,labels.size(1)):
            outputs, states = decoder_model(labels[:,t-1], features,states) 
            mask = labels[:,t:t+1]!=ds_test.vocab.token2id['_PAD_']
            outputs_ = mask*outputs
            labels_t = labels[:,t] * mask[:,0]
            loss = criterion(outputs, labels_t)
            total_loss += loss 
            predict = torch.unsqueeze(torch.argmax(outputs,-1), -1)
            predictions.append(predict)
        predictions = torch.cat(predictions,-1) 
        total_loss = total_loss/labels.size(1)
        return predictions, total_loss 





    def evaluate(encoder_model,decoder_model,testloader):
        encoder_model.eval()
        decoder_model.eval()
        gt_path = 'gt_data.txt'
        pred_path = 'predictions.txt'
        prediction_file = open(pred_path,'w')
        gt_file = open(gt_path,'w')
        with torch.no_grad():
            data_loader = tqdm(testloader)
            mean_loss = 0
            mean_acc = 0
            
            for data in data_loader:
                images, labels = data
                labels = labels[1]
                images = images.to(torch.float).to(device)
                labels = labels.to(device)
                outputs, loss = forward_pass(images, labels)
                acc = (outputs == labels[:,1:]).sum()/(labels.size(0)*outputs.size(1))
                outputs = outputs.cpu().numpy()
                for i in range(outputs.shape[0]):
                    seq = list(outputs[i])
                    pred_text = ds_test.vocab.seq2text(seq)
                    prediction_file.write(pred_text)
                    prediction_file.write('\n') 
                    gt_seq = list(labels[i].cpu().numpy())
                    gt_text = ds_test.vocab.seq2text(gt_seq)
                    gt_file.write(gt_text)
                    gt_file.write('\n')

                mean_loss += loss.item()
                mean_acc += acc.item()

        prediction_file.close()
        gt_file.close()
        metrics = score_files(gt_path, pred_path)
        mean_loss /= len(testloader)
        mean_acc /= len(testloader)
        return mean_loss, mean_acc, metrics



    test_loss, test_accuracy, metrics = evaluate(encoder_model, decoder_model, test_loader)
    print(" test , loss = %2.4f  accuracy = %2.4f"%(test_loss, test_accuracy) ) 
    print()
    print(metrics)



if __name__ == '__main__':
    model_path = '/media/altex/XcDrive/projects/im2latex-pytorch/ckpt-modelV0_1-best.pt'
    test(model_path)