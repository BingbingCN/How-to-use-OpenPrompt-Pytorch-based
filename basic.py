#!/usr/bin/env python
# coding: utf-8

# In[29]:


from datasets import load_dataset
from openprompt.data_utils import InputExample
from openprompt.plms import load_plm
from openprompt.prompts import ManualTemplate,ManualVerbalizer
from openprompt import PromptDataLoader
from openprompt import PromptForClassification
from transformers import AdamW,get_linear_schedule_with_warmup
from torch.utils.tensorboard import SummaryWriter
from pprint import pprint
from tqdm import tqdm
import torch


print('Load Dataset..')
raw_dataset = load_dataset('super_glue', 'cb')
print(raw_dataset['train'][0])

dataset={}
for split in ['train', 'validation', 'test']:
    dataset[split] = []
    for data in raw_dataset[split]:
        input_example = InputExample(text_a = data['premise'], text_b = data['hypothesis'], label=int(data['label']), guid=data['idx'])
        dataset[split].append(input_example)
print(dataset['train'][0])


# In[ ]:


print('Load Model..')
plm, tokenizer, model_config, WrapperClass = load_plm("t5", "t5-base")
print('Build Template..')
template_text = '{"placeholder":"text_a"} Question: {"placeholder":"text_b"}? Is it correct? {"mask"}.'
mytemplate = ManualTemplate(tokenizer=tokenizer, text=template_text)
wrapped_example = mytemplate.wrap_one_example(dataset['train'][0])
'''
Return two list 1-template-text,2-label and guid
'''
pprint(wrapped_example)


# In[ ]:


# Note that when t5 is used for classification, we only need to pass <pad> <extra_id_0> <eos> to decoder.
# The loss is calcaluted at <extra_id_0>. Thus passing decoder_max_length=3 saves the space
wrapped_t5tokenizer=WrapperClass(max_seq_length=128,
                                 decoder_max_length=3,
                                 tokenizer=tokenizer,
                                 truncate_method='head')
tokenized_example=wrapped_t5tokenizer.tokenize_one_example(wrapped_example,teacher_forcing=False)
print(tokenized_example)


# In[ ]:


model_inputs = {}
for split in ['train', 'validation', 'test']:
    print('Processing: ',split.title())
    model_inputs[split] = []
    for sample in tqdm(dataset[split]):
        tokenized_example = wrapped_t5tokenizer.tokenize_one_example(mytemplate.wrap_one_example(sample), teacher_forcing=False)
        model_inputs[split].append(tokenized_example)


# In[ ]:


print('Build Dataloader..')
train_dataloader = PromptDataLoader(dataset=dataset["train"], template=mytemplate, tokenizer=tokenizer,
    tokenizer_wrapper_class=WrapperClass, max_seq_length=128, decoder_max_length=3,
    batch_size=4,shuffle=True, teacher_forcing=False, predict_eos_token=False,
    truncate_method="head")
valid_dataloader = PromptDataLoader(dataset=dataset["validation"], template=mytemplate, tokenizer=tokenizer,
    tokenizer_wrapper_class=WrapperClass, max_seq_length=128, decoder_max_length=3,
    batch_size=4,shuffle=True, teacher_forcing=False, predict_eos_token=False,
    truncate_method="head")
test_dataloader = PromptDataLoader(dataset=dataset["test"], template=mytemplate, tokenizer=tokenizer,
    tokenizer_wrapper_class=WrapperClass, max_seq_length=128, decoder_max_length=3,
    batch_size=4,shuffle=True, teacher_forcing=False, predict_eos_token=False,
    truncate_method="head")


# In[ ]:


myverbalizer = ManualVerbalizer(tokenizer, num_classes=3,
                        label_words=[["yes"], ["no"], ["maybe"]])
print(myverbalizer.label_words_ids)
logits = torch.randn(2,len(tokenizer)) # creating a pseudo output from the plm, and
print(myverbalizer.process_logits(logits))


# In[ ]:


PromptModel=PromptForClassification(plm=plm,template=mytemplate,verbalizer=myverbalizer,freeze_plm=False)
PromptModel=PromptModel.cuda()
loss_func=torch.nn.CrossEntropyLoss()
# it's always good practice to set no decay to biase and LayerNorm parameters
no_decay = ['bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in PromptModel.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in PromptModel.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]
optimizer=torch.optim.AdamW(optimizer_grouped_parameters,lr=1.5e-4)
writer=SummaryWriter(log_dir='log_dir')


# In[ ]:

#
best_loss=9999999
for epoch in range(10):
    train_loss = 0
    PromptModel.train()
    par=tqdm(train_dataloader)
    for step, inputs in enumerate(par):
        inputs.cuda()
        logits = PromptModel(inputs)
        labels = inputs['label']
        loss = loss_func(logits, labels)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        optimizer.zero_grad()
        par.set_description('epoch:{}'.format(epoch+1))
        par.set_postfix(loss=loss.item())
    writer.add_scalars('Loss',{'train':train_loss/len(train_dataloader)},epoch+1)
    PromptModel.eval()
    val_loss=0
    par=tqdm(valid_dataloader)
    for step, inputs in enumerate(par):
        inputs.cuda()
        logits = PromptModel(inputs)
        labels = inputs['label']
        loss = loss_func(logits, labels)
        val_loss += loss.item()
        par.set_description('epoch:{}'.format(epoch+1))
        par.set_postfix(loss=loss.item())
    val_loss=val_loss/len(valid_dataloader)
    writer.add_scalars('Loss',{'valid':val_loss},epoch+1)

    if val_loss<best_loss:
        best_loss=val_loss
        print('Save Model..')
        torch.save(PromptModel.state_dict(),'t5-base-prompt.pt')


# In[ ]:


print('Load Best Model..')
PromptModel.load_state_dict(torch.load('t5-base-prompt.pt'))


# In[ ]:


allpreds = []
alllabels = []
for step, inputs in enumerate(valid_dataloader):

    inputs = inputs.cuda()
    logits = PromptModel(inputs)
    labels = inputs['label']
    alllabels.extend(labels.cpu().tolist())
    allpreds.extend(torch.argmax(logits, dim=-1).cpu().tolist())

acc = sum([int(i==j) for i,j in zip(allpreds, alllabels)])/len(allpreds)
print('Best Accuracy:{:.2f}%'.format(acc*100))


# In[ ]:




