from numpy import array
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
#pip install tensorflow
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense,LSTM

# split a univariate sequence into samples
def split_sequence(sequence, n_steps):
	X, y = list(), list()
	for i in range(len(sequence)):
		# find the end of this pattern
		end_ix = i + n_steps
		# check if we are beyond the sequence
		if end_ix > len(sequence)-1:
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
		X.append(seq_x)
		y.append(seq_y)
	return array(X), array(y)
 
# define input sequence
raw_seq = [110,125,133,146,158,172,187,196,210]
# choose a number of time steps
n_steps = 3
# split into samples
X, y = split_sequence(raw_seq, n_steps)
# reshape from [samples, timesteps] into [samples, timesteps, features]
n_features = 1
X = X.reshape((X.shape[0], X.shape[1], n_features))


# define model
model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(n_steps, n_features)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')


# fit model
model.fit(X, y, epochs=200, verbose=0)


# demonstrate prediction
x_input = array([187,196,210])
temp_input=list(x_input)
output=[]
i=0

while(i,10):
    if(len(temp_input)>3):
        x_input=array(temp_input[1:])
        x_input=x_input.reshape(1,n_steps,n_features)
        yhat=model.predict(x_input,verbose=0)
        temp_input.append(yhat[0][0])
        temp_input=temp_input[1:]
        output.append(yhat[0][0])
        i=i+1
        break
    else:
        x_input=x_input.reshape(1,n_steps,n_features)
        yhat=model.predict(x_input,verbose=0)
        temp_input.append(yhat[0][0])
        output.append(yhat[0][0])
        i=i+1
        break
print(output)
