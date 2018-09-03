# -*- coding: utf-8 -*-
import random
import gensim
import gzip
import numpy as np
import ast
import copy
import sys
from sklearn.model_selection  import train_test_split
from prettytable import PrettyTable
import re
import tensorflow as tf

"""Generic set of classes and methods"""


def strToLst(string):
    return ast.literal_eval(string)


class HeadData:
    def __init__(self, data, indices):
        self.data = data
        self.indices = indices

    def split(self, fraction):

        data_train, data_test, idx_train, idx_test = train_test_split(self.data, self.indices, test_size=fraction,
                                                                      random_state=42)

        train = HeadData(data_train, idx_train)

        test = HeadData(data_test, idx_test)
        return train, test

def transformToInitialInput(matrix,tags):
        active_relations = np.nonzero(matrix)
        active_relations_iidx = active_relations[0]
        active_relations_jidx = active_relations[1]

        tokens_ids = []
        heads_ids = []
        labels_ids = []
        head_labels_ids = []
        labels_name = []

        for m_idx in range(len(matrix)):
            tokens_ids.append(m_idx)
            heads_ids.append([])
            labels_ids.append([])
            head_labels_ids.append([])
            labels_name.append([])

        for i_idx in range(len(active_relations_iidx)):
            head_id = int(active_relations_jidx[i_idx] / len(tags))
            label_id = active_relations_jidx[i_idx] % len(tags)
            token_id = active_relations_iidx[i_idx]
            head_label_id = active_relations_jidx[i_idx]

            # idx=tokens_ids.index(token_id)
            heads_ids[token_id].append(head_id)
            labels_ids[token_id].append(label_id)
            head_labels_ids[token_id].append(head_label_id)
            labels_name[token_id].append(tags[label_id])

            # print (str(token_id) + " " +str(head_label_id)+ " " +str(head)+ " " +str(label))
        return tokens_ids, head_labels_ids, labels_ids, heads_ids, labels_name


###run one time to obtain the characters
def getCharsFromDocuments(documents):
    chars = []
    for doc in documents:
        for tokens in doc.tokens:
            for char in tokens:
                # print (token)
                chars.append(char)
    chars = list(set(chars))
    chars.sort()
    return chars


###run one time to obtain the ner labels
def getEntitiesFromDocuments(documents):
    BIOtags = []
    ECtags = []
    for doc in documents:
        for tag in doc.BIOs:
            BIOtags.append(tag)
            if tag.startswith("B-") or tag.startswith("I-"):
                ECtags.append(tag[2:])
            else:
                ECtags.append(tag)

    BIOtags = list(set(BIOtags))
    BIOtags.sort()
    ECtags = list(set(ECtags))
    ECtags.sort()
    return BIOtags, ECtags


def getECfromBIO(BIO_tag):
    if BIO_tag.startswith("B-") or BIO_tag.startswith("I-"):
        return (BIO_tag[2:])
    else:
        return (BIO_tag)


###run one time to obtain the relations
def getRelationsFromDocuments(documents):
    relations = []
    for doc in documents:
        for relation_list in doc.relations:
            for relation in relation_list:
                relations.append(relation)

    relations = list(set(relations))
    relations.sort()
    return relations


def tokenToCharIds(token, characters):
    charIds = []
    for char in token:
        charIds.append(characters.index(char))
    return charIds


def labelsListToIds(listofLabels, setofLabels):
    labelIds = []
    for label in listofLabels:
        labelIds.append(setofLabels.index(label))

    return labelIds


def getScoringMatrixHeads(listofRelations, setofLabels, heads):
    scoringMatrixHeads = []
    relationIds = labelsListToIds(listofRelations, setofLabels)


    for relIdx in range(len(relationIds)):
        # print (rels[relIdx]*getNumberOfClasses()+labelJointIds[relIdx])
        scoringMatrixHeads.append(heads[relIdx] * len(setofLabels) + relationIds[relIdx])
    return scoringMatrixHeads


def getLabelId(label, setofLabels):
    return setofLabels.index(label)

def strToBool(str):
    if str.lower() in ['true', '1']:
        return True
    return False



def getEmbeddingId(word, embeddingsList):
    # modified method from http://cistern.cis.lmu.de/globalNormalization/globalNormalization_all.zip
    if word != "<empty>":
        if not word in embeddingsList:
            if re.search(r'^\d+$', word):
                word = "0"
            if word.islower():
                word = word.title()
            else:
                word = word.lower()
        if not word in embeddingsList:
            word = "<unk>"
        curIndex = embeddingsList[word]
        return curIndex


def readWordvectorsNumpy(wordvectorfile, isBinary=False):

    # modified method from http://cistern.cis.lmu.de/globalNormalization/globalNormalization_all.zip
    wordvectors = []
    words = []
    model = gensim.models.KeyedVectors.load_word2vec_format(wordvectorfile, binary=isBinary,unicode_errors='ignore')

    vectorsize = model.vector_size

    for key in list(model.vocab.keys()):
        wordvectors.append(model.wv[key])
        words.append(key)

    zeroVec = [0 for i in range(vectorsize)]
    random.seed(123456)
    randomVec = [random.uniform(-np.sqrt(1. / len(wordvectors)), np.sqrt(1. / len(wordvectors))) for i in
                 range(vectorsize)]
    wordvectors.insert(0, randomVec)
    words.insert(0, "<unk>")
    wordvectors.insert(0, zeroVec)
    words.insert(0, "<empty>")

    wordvectorsNumpy = np.array(wordvectors)
    return wordvectorsNumpy, vectorsize, words


def readIndices(wordvectorfile, isBinary=False):
    # modified method from http://cistern.cis.lmu.de/globalNormalization/globalNormalization_all.zip
    indices = {}
    curIndex = 0
    indices["<empty>"] = curIndex
    curIndex += 1
    indices["<unk>"] = curIndex
    curIndex += 1

    model = gensim.models.KeyedVectors.load_word2vec_format(wordvectorfile, binary=isBinary,unicode_errors='ignore')

    count = 0
    # c=0
    for key in list(model.vocab.keys()):
        indices[key] = curIndex
        curIndex += 1

    return indices



def printParameters(config):

    t = PrettyTable(['Params', 'Value'])

    #dataset
    t.add_row(['Config', config.config_fname])
    t.add_row(['Embeddings', config.filename_embeddings])
    t.add_row(['Embeddings size ', config.representationsize])
    t.add_row(['Train', config.filename_train])
    t.add_row(['Dev', config.filename_dev])
    t.add_row(['Test', config.filename_test])

    #training
    t.add_row(['Epochs ', config.nepochs])
    t.add_row(['Optimizer ', config.optimizer])
    t.add_row(['Activation ', config.activation])
    t.add_row(['Learning rate ', config.learning_rate])
    t.add_row(['Gradient clipping ', config.gradientClipping])
    t.add_row(['Patience ', config.nepoch_no_imprv])
    t.add_row(['Use dropout', config.use_dropout])
    t.add_row(['Ner loss ', config.ner_loss])
    t.add_row(['Ner classes ', config.ner_classes])
    t.add_row(['Use char embeddings ', config.use_chars])
    t.add_row(['Use adversarial',config.use_adversarial])

    # hyperparameters
    t.add_row(['Dropout embedding ', config.dropout_embedding])
    t.add_row(['Dropout lstm ', config.dropout_lstm])
    t.add_row(['Dropout lstm output ', config.dropout_lstm_output])
    t.add_row(['Dropout fcl ner ', config.dropout_fcl_ner])
    t.add_row(['Dropout fcl rel ', config.dropout_fcl_rel])
    t.add_row(['Hidden lstm size ', config.hidden_size_lstm])
    t.add_row(['LSTM layers ', config.num_lstm_layers])
    t.add_row(['Hidden nn size ', config.hidden_size_n1])
    t.add_row(['Char embeddings size ', config.char_embeddings_size])
    t.add_row(['Hidden size char ', config.hidden_size_char])
    t.add_row(['Label embeddings size ', config.label_embeddings_size])
    t.add_row(['Alpha ', config.alpha])

    #evaluation
    t.add_row(['Evaluation method ', config.evaluation_method])


    print(t)

def getSegmentationDict(lst):
    return {k: v for v, k in enumerate(lst)}

def generator(data, m,config,train=False):
    # generate the data
    embeddingIds = m['embeddingIds']
    isTrain=m['isTrain']

    scoringMatrixGold = m['scoringMatrixGold']
    BIO = m['BIO'] # always the BIO tags
    entity_tags=m['entity_tags'] # either the BIO tags or the EC tags - depends on the NER target values
    entity_tags_ids = m['entity_tags_ids']
    tokens = m['tokens']
    tokenIds = m['tokenIds']
    charIds = m['charIds']
    tokensLens = m['tokensLens']

    seqlen = m['seqlen']
    doc_ids=m['doc_ids']


    dropout_embedding_keep = m['dropout_embedding']
    dropout_lstm_keep = m['dropout_lstm']
    dropout_lstm_output_keep = m['dropout_lstm_output']
    dropout_fcl_ner_keep = m['dropout_fcl_ner']
    dropout_fcl_rel_keep = m['dropout_fcl_rel']


    dropout_embedding_prob = 1
    dropout_lstm_prob = 1
    dropout_lstm_output_prob = 1
    dropout_fcl_ner_prob = 1
    dropout_fcl_rel_prob = 1

    if config.use_dropout == True and train==True:

        dropout_embedding_prob = config.dropout_embedding
        dropout_lstm_prob = config.dropout_lstm
        dropout_lstm_output_prob = config.dropout_lstm_output
        dropout_fcl_ner_prob = config.dropout_fcl_ner
        dropout_fcl_rel_prob = config.dropout_fcl_rel

    data_copy = copy.deepcopy(data)
    # train_ind=np.arange(len(train.data))
    if config.shuffle == True:
        shuffled_data, _, shuffled_data_idx, _ = train_test_split(data_copy.data, data_copy.indices, test_size=0,
                                                                  random_state=42)
        # shuffled_data, _, shuffled_data_idx, _ = train_test_split(data_copy.data, data_copy.indices, test_size=0,random_state=42)

        data_copy = HeadData(shuffled_data, shuffled_data_idx)
        # print ("shuffle:"+ str(shuffle) )
        # print(data_copy.indices)
    else:

        data_copy = HeadData(data_copy.data, data_copy.indices)
        # data_copy = HeadData(data_copy.data, data_copy.indices)

        # print("shuffle:" + str(shuffle))
        # print(data_copy.indices)

    # batchsize=16 # number of documents per batch
    batches_embeddingIds = []  # e.g., 131 batches
    batches_charIds = []  # e.g., 131 batches
    batches_scoringMatrixHeadIds = []  # e.g., 131 batches
    batches_scoringMatrix = []  # e.g., 131 batches
    batches_tokens = []

    batches_entity_tags = []
    batches_entity_tags_ids = []
    batches_BIO=[]
    batches_tokenIds = []
    batches_doc_ids = []

    docs_batch_embeddingIds = []  # e.g., 587 max doc length - complete with -1 when the size of the doc is smaller
    docs_batch_charIds = []  # e.g., 587 max doc length - complete with -1 when the size of the doc is smaller
    docs_batch_scoringMatrixHeadIds = []
    docs_batch_scoringMatrix = []

    docs_batch_entity_tags=[] 
    docs_batch_entity_tags_ids = []

    docs_batch_tokens = []

    docs_batch_BIO = []
    docs_batch_tokenIds = []
    docs_batch_doc_ids = []

    maxDocLenList = []
    maxSentenceLen = -1

    maxWordLenList = []
    maxWordLen = -1

    wordLenList = []
    wordLens = []

    lenBatchesDoc = []
    lenEmbeddingssDoc = []

    lenBatchesChars = []
    lenCharsDoc = []

    sumLen = 0
    for docIdx in range(len(data_copy.data)):
        doc = data_copy.data[docIdx]
        # print (doc)
        if docIdx % config.batchsize == 0 and docIdx > 0:
            # print (docIdx)
            # print ("new batch")
            batches_embeddingIds.append(docs_batch_embeddingIds)
            batches_charIds.append(docs_batch_charIds)

            batches_scoringMatrixHeadIds.append(docs_batch_scoringMatrixHeadIds)
            batches_scoringMatrix.append(docs_batch_scoringMatrix)
            batches_entity_tags.append(docs_batch_entity_tags)
            batches_entity_tags_ids.append(docs_batch_entity_tags_ids)

            batches_tokens.append(docs_batch_tokens)

            batches_BIO.append(docs_batch_BIO)
            batches_tokenIds.append(docs_batch_tokenIds)
            batches_doc_ids.append(docs_batch_doc_ids)

            docs_batch_embeddingIds = []  # e.g., 587 max doc length - complete with -1 when the size of the doc is smaller
            docs_batch_charIds = []  # e.g., 587 max doc length - complete with -1 when the size of the doc is smaller
            docs_batch_scoringMatrixHeadIds = []
            docs_batch_scoringMatrix = []

            docs_batch_tokens = []

            docs_batch_entity_tags = []
            docs_batch_entity_tags_ids = []
            docs_batch_BIO = []
            docs_batch_tokenIds = []
            docs_batch_doc_ids = []

            maxDocLenList.append(maxSentenceLen)
            maxSentenceLen = -1

            maxWordLenList.append(maxWordLen)
            maxWordLen = -1

            wordLenList.append(wordLens)



        if len(doc.token_ids) > maxSentenceLen:
            maxSentenceLen = len(doc.token_ids)

        longest_token_list=max(doc.char_ids, key=len)
        if len(longest_token_list) > maxWordLen:
            maxWordLen = len(longest_token_list)

        wordLens=[len(token) for token in doc.char_ids]


        sumLen += len(doc.token_ids)
        docs_batch_embeddingIds.append(doc.embedding_ids)
        docs_batch_charIds.append(doc.char_ids)
        docs_batch_scoringMatrixHeadIds.append(doc.joint_ids)

        scoringMatrix = np.zeros((len(doc.joint_ids), len(doc.joint_ids) *len(config.dataset_set_relations) ))

        for tokenIdx in range(len(doc.joint_ids)):
            tokenHeads = doc.joint_ids[tokenIdx]
            for head in tokenHeads:
                # print (str(tokenIdx)+ " "+ str(head))
                scoringMatrix[tokenIdx, head] = 1

        docs_batch_scoringMatrix.append(scoringMatrix)
        # print (scoringMatrix)

        #print (doc.jlabel_names)
        if config.ner_classes=="BIO":
            docs_batch_entity_tags.append(doc.BIOs)##to do
            docs_batch_entity_tags_ids.append(doc.BIO_ids)

        elif config.ner_classes=="EC":
            docs_batch_entity_tags.append(doc.ecs)##to do
            docs_batch_entity_tags_ids.append(doc.ec_ids)

        docs_batch_tokens.append(doc.tokens)

        docs_batch_BIO.append(doc.BIOs)##to do
        docs_batch_tokenIds.append(doc.token_ids)
        docs_batch_doc_ids.append(doc.docId)
        if docIdx == len(
                data_copy.data) - 1:  ## if there are no documents left - append the batch - usually it is shorter batch
            batches_embeddingIds.append(docs_batch_embeddingIds)
            batches_charIds.append(docs_batch_charIds)
            batches_scoringMatrixHeadIds.append(docs_batch_scoringMatrixHeadIds)
            batches_scoringMatrix.append(docs_batch_scoringMatrix)

            batches_entity_tags.append(docs_batch_entity_tags)
            batches_entity_tags_ids.append(docs_batch_entity_tags_ids)
            batches_tokens.append(docs_batch_tokens)

            batches_BIO.append(docs_batch_BIO)
            batches_tokenIds.append(docs_batch_tokenIds)
            batches_doc_ids.append(docs_batch_doc_ids)
            maxDocLenList.append(maxSentenceLen)
            maxWordLenList.append(maxWordLen)
            wordLenList.append(wordLens)
            # maxDocLen.append(maxWordLen)

    # print(maxDocLen)
    for bIdx in range(len(batches_embeddingIds)):

        batch_embeddingIds = batches_embeddingIds[bIdx]
        batch_charIds = batches_charIds[bIdx]
        batch_scoringMatrixHeadIds = batches_scoringMatrixHeadIds[bIdx]

        batch_entity_tags = batches_entity_tags[bIdx]
        batch_tokens = batches_tokens[bIdx]

        batch_tokenIds = batches_tokenIds[bIdx]

        for dIdx in range(len(batch_embeddingIds)):
            embeddingId_doc = batch_embeddingIds[dIdx]
            charIds_doc = batch_charIds[dIdx]
            scoringMatrixHeadId_doc = batch_scoringMatrixHeadIds[dIdx]

            ner_doc=batch_entity_tags[dIdx]
            token_doc = batch_tokens[dIdx]

            token_id_doc = batch_tokenIds[dIdx]

            lenEmbeddingssDoc.append(len(embeddingId_doc))
            tokensLen=[len(token) for token in charIds_doc]
            lenCharsDoc.append(tokensLen)


            for tokenIdx in range(len(tokensLen)):
                tokenLen=tokensLen[tokenIdx]

                if tokenLen<maxWordLenList[bIdx]:

                    for i in np.arange(maxWordLenList[bIdx]-tokenLen):
                        #print (charIds_doc)
                        charIds_doc[tokenIdx].append(0)


            if len(embeddingId_doc) < maxDocLenList[bIdx]:
                # print  (maxWordLen-len(word_doc))
                # print ('here')
                for i in np.arange(maxDocLenList[bIdx] - len(embeddingId_doc)):
                    # pass
                    embeddingId_doc.append(0)
                    charIds_doc.append([])


                    scoringMatrixHeadId_doc.append([maxDocLenList[bIdx] - 1])
                    token_doc.append("ZERO")

                    ner_doc.append("ZERO")
                    token_id_doc.append(maxDocLenList[bIdx] - 1)

        lenBatchesDoc.append(lenEmbeddingssDoc)

        lenBatchesChars.append(lenCharsDoc)

        lenEmbeddingssDoc = []
        lenCharsDoc=[]

    # return batches_words,batches_heads
    for bIdx in range(len(batches_embeddingIds)):  # 131
        # print (bIdx)
        batch_embeddingIds = np.asarray(batches_embeddingIds[bIdx])
        batch_charIds = np.asarray(batches_charIds[bIdx])
        batch_scoringMatrix = np.asarray(batches_scoringMatrix[bIdx])

        batch_ner = np.asarray(batches_entity_tags[bIdx])
        batch_ner_ids = np.asarray(batches_entity_tags_ids[bIdx])
        batch_token = np.asarray(batches_tokens[bIdx])

        batch_bio = np.asarray(batches_BIO[bIdx])
        batch_tokenId = np.asarray(batches_tokenIds[bIdx])

        batch_doc_id = np.asarray(batches_doc_ids[bIdx])

        docs_length = np.asarray(lenBatchesDoc[bIdx])
        tokenslength = np.asarray(lenBatchesChars[bIdx])




        yield {dropout_embedding_keep:dropout_embedding_prob,dropout_lstm_keep:dropout_lstm_prob,dropout_lstm_output_keep:dropout_lstm_output_prob,
               dropout_fcl_ner_keep:dropout_fcl_ner_prob,dropout_fcl_rel_keep:dropout_fcl_rel_prob,isTrain:train,charIds:batch_charIds,
               tokensLens:tokenslength, embeddingIds: batch_embeddingIds,entity_tags_ids:batch_ner_ids,entity_tags:batch_ner,
               tokens:batch_token,BIO: batch_bio,tokenIds:batch_tokenId,scoringMatrixGold:batch_scoringMatrix, seqlen:docs_length, doc_ids:batch_doc_id }





