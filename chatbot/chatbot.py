"""This is the chatbot using pandas or may be tensorflow 2.0
with 3 layer and 24 node of neura network with own practice and knowledge 
and data is taken and cleaned using sentdex tutrial"""

import numpy as np
import tensorflow as tf
import pandas as pd
import re
import time






























# import numpy as np
# import tensorflow as tf
# import re
# import time


# # importing dataset
# lines = open('movie_lines.txt', encoding='utf-8', errors='ignore').read().split('\n')
# conversations = open('movie_conversations.txt', encoding='utf-8', errors='ignore').read().split('\n')

# #create the dictionary taht maps each line and its id
# id2line = {}
# for line in lines:
#     _line= line.split(' +++$+++ ')
#     if len(_line) == 5:
#         id2line[_line[0]] = _line[4]
# # print(id2line)
# conversations_ids = []
# for conversation in conversations[:-1]:
#     _conversation = conversation.split(' +++$+++ ')[-1][1:-1].replace("'","").replace(' ','')
#     conversations_ids.append(_conversation.split(','))


# questions = []
# answers = []

# for conversation in conversations_ids:
#     for i in range(len(conversation)-1):
#         questions.append(id2line[conversation[i]])
#         answers.append(id2line[conversation[i+1]])


# def clean_text(text):
#     text = text.lower()
#     text = re.sub(r"i'm", "i am", text)
#     text = re.sub(r"he's", "he is", text)
#     text = re.sub(r"she's", "she is", text)
#     text = re.sub(r"that's", "that is", text)
#     text = re.sub(r"what's", "what is", text)
#     text = re.sub(r"where's", "where is", text)
#     text = re.sub(r"\'ll", "will", text)
#     text = re.sub(r"\ve", "have", text)
#     text = re.sub(r"\'re", "are", text)
#     text = re.sub(r"\'d", "would", text)
#     text = re.sub(r"won't", "will not", text)
#     text = re.sub(r"can't", "cannot", text)
#     text = re.sub(r"[-()\"#/@;:{}+=~|.?,]", "", text)
#     return text


# #cleaning the questions
# clean_questions = []
# for question in questions:
#     clean_questions.append(clean_text(question))



# #cleaning the answers
# clean_answers = []
# for answer in answers:
#     clean_answers.append(clean_text(question))

# #remove recorrence of world
# word2count = {}
# for question in clean_questions:
#     for word in question.split():
#         if word not in word2count:
#             word2count[word] = 1
#         else:
#             word2count[word] += 1

# for answer in clean_answers:
#     for word in answer.split():
#         if word not in word2count:
#             word2count[word] = 1
#         else:
#             word2count[word] += 1


# threshold = 20
# questionworlds2int = {}
# word_number = 0
# for word, count in word2count.items():
#     if count >= threshold:
#         questionworlds2int[word] = word_number
#         word_number+= 1

# answersworlds2int = {}
# word_number = 0
# for word, count in word2count.items():
#     if count >= threshold:
#         answersworlds2int[word] = word_number
#         word_number+= 1

# #adding the tokens
# tokens = ['<PAD>', '<EOS>', '<OUT>', '<SOS>']
# for token in tokens:
#     questionworlds2int[token] = len(questionworlds2int)+1

# for token in tokens:
#    answersworlds2int[token] = len(answersworlds2int)+1

# #creating the inverse dictionary of the answerswords2int dict
# answersints2word = { w_i: w for w, w_i in answersworlds2int.items() }

# #adding the EOS token to the answers list
# for i in range(len(clean_answers)):
#     clean_answers[i] += ' <EOS>'

# questions_to_int = []
# for question in clean_questions:
#     ints = []
#     for world in question.split():
#         if word not in questionworlds2int:
#             ints.append(questionworlds2int['<OUT>'])
#         else:
#             ints.append(questionworlds2int[word])
#     questions_to_int.append(ints)

# answers_to_int = []
# for answer in clean_answers:
#     ints = []
#     for world in answer.split():
#         if word not in answersworlds2int :
#             ints.append(answersworlds2int['<OUT>'])
#         else:
#             ints.append(answersworlds2int[word])
#     answers_to_int.append(ints)

# #sorting questions and answers by the length of questions
# sorted_clean_questions = []
# sorted_clean_answers = []
# for length in range(1,25):
#     for i in enumerate(questions_to_int):
#         if len(i[1])== length:
#             sorted_clean_questions.append(questions_to_int[i[0]])
#             sorted_clean_answers.append(answers_to_int[i[0]])


# # creating placeholders for the inputs and the target
# def model_inputs():
#     inputs = tf.placeholder(tf.int32, [None, None], name='input')
#     targets = tf.placeholder(tf.int32, [None, None], name='target')
#     lr = tf.placeholder(tf.float32, name='learnng_rate')
#     keep_prob = tf.placeholder(tf.float32, name='keep_prob')
#     return inputs, targets, lr, keep_prob

# #preprocessing the target
# def preprocess_targets(targets, word2int, batch_size):
#     left_side = tf.fill([batch_size, 1], word2int['<SOS>'])
#     right_side = td.strided_slice(targets, [0,0], [batch_size, -1], [1,1])
#     preprocessed_targets = tf.concat([left_side, right_side], 1)
#     return preprocessed_targets

# #creating the encoder RNN layer
# def encoder_rnn(rnn_inputs, rnn_size, num_layers, keep_prob, sequence_length):
#     lstm = tf.contrib.rnn.BasicLSTMCell(rnn_size)
#     lstm_dropout = tf.contrib.DropoutWrapper(lstm, input_keep_prob = keep_prob)
#     encoder_cell = tf.contrib.rnn.MultiRNNCell([lstm_dropout * num_layers])
#     _, encoder_state = tf.nn.bidirectional_dynamic_rnn(cell_fw = encoder_cell, cell_bw = encoder_cell,
#         sequence_length = sequence_length, inputs = rnn_inputs, dtype = tf.float32)
#     return encoder_state


# #decoding the training set
# def decode_training_set(encoder_state, decoder_cell, decoder_embeded_input, sequence_length, decoding_scope,
#  output_funtion, keep_prob, batch_size):
#     attention_states = tf.zeros([batch_size, 1, decoder_cell.output_size])
#     attention_keys, attention_values, attention_score_function, attention_construct_function = tf.contrib.seq2seq.prepare_attention(attention_states,
#         attention_options = 'bahdanu', no_units = decoder_cell.output_size)
#     training_decoder_function = tf.contrib.seq2seq.attention_decoder_fn_train(encoder_state[0],
#         attention_keys,
#         attention_values,
#         attention_score_function,
#         attention_construct_function,
#         name = 'attn_dec_train')
#     decoder_output, _, _ = tf.contrib.seq2seq.dynamic_rnn_decoder(decoder_cell,
#         training_decoder_function,
#         decoder_embeded_input,
#         sequence_length,
#         scope = decoding_scope)  # decoder_output, decoder_final_state, decode_final_context_state
#     decoder_output_dropout = tf.nn.dropout(decoder_output, keep_prob)
#     return output_funtion(decoder_output_dropout)

# #decoding the text/validation set
# def decode_test_set(encoder_state, decoder_cell, decoder_embeddings_matrix,sos_id, eos_id, maximum_lenth, num_words, sequence_length, decoding_scope,
#  output_funtion, keep_prob, batch_size):
#     attention_states = tf.zeros([batch_size, 1, decoder_cell.output_size])
#     attention_keys, attention_values, attention_score_function, attention_construct_function = tf.contrib.seq2seq.prepare_attention(attention_states,
#         attention_options = 'bahdanu', no_units = decoder_cell.output_size)
#     test_decoder_function = tf.contraib.seq2seq.attention_decoder_fn_inference(output_function,
#         encoder_state[0],
#         attention_keys,
#         attention_values,
#         attention_score_function,
#         attention_construct_function,
#         decoder_embeddings_matrix,
#         sos_id,
#         eos_id,
#         maximum_lenth,
#         num_words,
#         name = 'attn_dec_inf')
#     test_predictions, _, _ = tf.contrib.seq2seq.dynamic_rnn_decoder(decoder_cell,
#         test_decoder_function,
#         scope = decoding_scope)  # decoder_output, decoder_final_state, decode_final_context_state
#     return test_predictions

# # creating docder RNN
# def decoder_rnn(decoder_embeded_input, decoder_embeddings_matrix, encoder_state, num_words,
#     sequence_length, rnn_size, num_layers, word2int, keep_prob, batch_size):
#     with tf.variable_scope('decoding') as decoding_scope:
#         lstm = tf.contrib.rnn.BasicLSTMCell(rnn_size)
#         lstm_dropout = tf.contrib.DropoutWrapper(lstm, input_keep_prob = keep_prob)
#         decoder_cell = tf.contrib.rnn.MultiRNNCell([lstm_dropout] * num_layers)
#         weights = tf.truncated_normal_initializer(stddev = 0.1)
#         bioses = tf.zeros_initializer()
#         output_funtion = lambda x: tf.contrib.layers.fully_connected(x, num_words, None, scope = decoding_scope, weights_initializers = weights, bioses_initializer = bioses)

#         training_predictions = decode_training_set(encoder_state, decoder_cell, decoder_embeded_input, sequence_length, decoding_scope, output_funtion, keep_prob, batch_size)

#         decoding_scope.reuse_variables()
#         test_predictions = decode_test_set(encoder_state, decoder_cell, decoder_embeddings_matrix, word2int['<SOS>'], word2int['<EOS>'], sequence_length-1, num_words, decoding_scope, output_funtion, keep_prob, batch_size)

#     return training_predictions, test_predictions

# #building the seq2seq model
# def seq2seq_model(inputs, targets, keep_prob, batch_size, sequence_length, answers_num_words, questions_num_worlds, encoder_embedding_size, decoder_embeddings_size, rnn_size, num_layers, questionworlds2int):
#     encoder_embeded_input = tf.contrib.layers.embed_sequence(inputs, answers_num_words+1,encoder_embedding_size, initilizer = tf.random_uniform_initializer(0, 1))
#     encoder_state = encoder_rnn(encoder_embeded_input, rnn_size, num_layers, keep_prob, sequence_length)
#     preprocess_targets = preprocess_targets(targets, questionworlds2int, batch_size)
#     decoder_embeddings_matrix = tf.variable(tf.random_uniform([questions_num_worlds+1, decoder_embeddings_size], 0, 1))
#     decoder_embeded_input = tf.nn.embedding_lookup(decoder_embeddings_matrix, preprocess_targets)
#     training_predictions, test_predictions = decoder_rnn(decoder_embeded_input, decoder_embeddings_matrix, encoder_state, questions_num_worlds, sequence_length, rnn_size, num_layers, questionworlds2int, keep_prob, batch_size)
#     return training_predictions, test_predictions


# #setting the Hypermeters (Training)
# epochs = 100 # if pc slow recude it
# batch_size = 64 # if pc slow 128 it
# rnn_size = 512
# num_layers = 3 #layer
# encoder_embedding_size = 512
# decoder_embedding_size = 512
# learnng_rate = 0.01
# learnng_rate_decay = 0.9
# min_leaning_rate = 0.0001
# keep_probability = 0.5

# #defining a tf session
# tf.reset_default_graph()
# session = tf.InteractiveSession()

# #loading the model inputs
# inputs, targets, lr, keep_prob = model_inputs()

# #setting the sequence length
# sequence_length = tf.placeholder_with_default(25, None, name = 'sequence_length')

# #getting hte shape of the input tensor
# input_shape = tf.shape(inputs)

# #getting the training and predictions
# training_predictions, test_predictions = seq2seq_model(tf.reverse(inputs, [-1]),
#     targets,
#     keep_prob,
#     batch_size,
#     sequence_length,
#     len(answers_num_words),
#     len(questions_num_worlds),
#     encoder_embedding_size,
#     decoder_embeddings_size,
#     rnn_size,
#     num_layers,
#     questionworlds2int)

# #setting up the loss error, the optimizer and grading clipping
# with tf.name_scope('optimization'):
#     loss_error = tf.contrib.seq2seq.sequence_loss(training_predictions, targets, tf.ones([input_shape[0], sequence_length]))
#     optimizer = tf.train.AdamOptimizer(learnng_rate)
#     gradients = optimizer.compute_gradients(loss_error)
#     clipped_gradients = [ (tf.clip_by_value(grad_tensor, -5. ,5.), grad_variable) for grad_tensor, grad_variable in gradients if grad_tensor is not None]
#     optimizer_gradient_clipping = optimizer.apply_gradients(clipped_gradients)


# # padding the sequence with the <PAD> token
# # to make the same length in question and answer
# def apply_padding(batch_of_sequences, word2int):
#     max_sequence_length = max([len(sequence) for sequence in batch_of_sequences])
#     return [ sequence + [word2int['<PAD>']] * (max_sequence_length - len(sequence)) for sequence in batch_of_sequences]

# # splitting the data into batches of questions and answers
# def split_into_batches(questions, answers, batch_size):
#     for batch_index in range(0, len(questions) // batch_size):
#         start_index =batch_index * batch_size
#         questions_in_batch = questions[start_index:start_index+batch_size]
#         answers_in_batch = answers[start_index:start_index+batch_size]
#         padded_questions_in_batch = np.array(apply_padding(questions_in_batch, questionworlds2int))
#         padded_answers_in_batch = np.array(apply_padding(answers_in_batch, answersworlds2int))
#         yield padded_questions_in_batch, padded_answers_in_batch


# # splitting the questiona and answers into training and validation sets
# training_validation_split = int(len(sorted_clean_questions) * 0.15)
# training_questions = sorted_clean_questions[training_validation_split:]
# training_answers = sorted_clean_answerss[training_validation_split:]
# validation_questions = sorted_clean_questions[:training_validation_split]
# validation_answers = sorted_clean_answers[:training_validation_split]

# #tarining
# batch_index_checking_training_loss = 100
# batch_index_checking_validation_loss = (len(training_questions) // batch_size // 2) -1
# total_tarining_loss_error = 0
# list_vilidation_loss_error = []
# early_stoping_check = 0
# early_stoping_stop = 1000   #100
# checkpoint = 'chatbot_weithts.ckpt'
# session.run(tf.global_variable_initializer())
# for epoch in range(1, epoch+1):
#     for batch_index, (padded_questions_in_batch, padded_answers_in_batch) in enumerate(split_into_batches(training_questions, training_answers, batch_size)):
#         starting_time = time.time()
#         _, batch_training_loss_error = session.run([optimizer_gradient_clipping, loss_error], {inputs : padded_questions_in_batch,
#                                                                                             target : padded_answers_in_batch,
#                                                                                             lr : learnng_rate,
#                                                                                             sequence_length : padded_questions_in_batch.shape[1],
#                                                                                             keep_prob : keep_probability })
#         total_tarining_loss_error += batch_training_loss_error
#         ending_time = time.time()
#         batch_time = ending_time - starting_time
#         if atch_index % batch_index_checking_training_loss == 0:
#             print('Epoch:{:>3}/{}, Batch : {:>4}/{}, Training Loss Error : {:>6.3f}, Training time on 100 Batches : {:d} seconds'.format(epoch,epochs, batch_index, len(training_questions/batch_size),total_tarining_loss_error/batch_index_checking_training_loss, int(batch_time * batch_index_checking_training_loss)))

#             total_tarining_loss_error = 0
#         if batch_index % batch_index_checking_validation_loss == 0 and batch_index > 0:
#             total_validation_loss_error = 0
#             starting_time = time.time()
#             for batch_index_validation, (padded_questions_in_batch, padded_answers_in_batch) in enumerate(split_into_batches(validation_questions, validation_answers, batch_size)):
#                 batch_validation_loss_error = session.run( loss_error, {inputs : padded_questions_in_batch,
#                                                                                             target : padded_answers_in_batch,
#                                                                                             lr : learnng_rate,
#                                                                                             sequence_length : padded_questions_in_batch.shape[1],
#                                                                                             keep_prob : 1 })
#                 total_tarining_loss_error += batch_validation_loss_error
#             ending_time = time.time()
#             batch_time = ending_time - starting_time
#             average_validation_loss_error = total_validation_loss_error / (len(validation_questions / batch_size))
#             print('Validation Loss error : {:>6.3f}, batch validation time {:d} seconds'.format(average_validation_loss_error,int(batch_time)))
#             learnng_rate *= learnng_rate_decay
#             if learnng_rate < min_leaning_rate:
#                 learnng_rate = min_leaning_rate
#             list_vilidation_loss_error.append(average_validation_loss_error)
#             if average_validation_loss_error<= min(list_vilidation_loss_error):
#                 print('I speak better now!!!')
#                 early_stoping_check = 0
#                 saver = tf.train.Saver()
#                 saver.save(session, checkpoint)
#             else:
#                 print('Sorry i do not speak batter, i need more practice.')
#                 early_stoping_check += 1
#                 if early_stoping_check == early_stoping_stop:
#                     break
#         if early_stoping_check == early_stoping_stop:
#                     print('Sorry this the best i can do')
#                     break
# print('game over')


# # testing

