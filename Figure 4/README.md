# Example data
Ca_data_example.mat is an example of the calcium imaging recording data of one cell as well as relevant behavioral information such as detected lick times and trial outcome during task.
This is the data used for all analyses using calcium imaging data.
eg_Data{n,1} = recorded dF/F0 (in sample number)
eg_Data{n,2} = Lick times (s)
eg_Data{n,3} = Behavior data (see below)
- 1st column : stim onset time (in sample number)
- 2nd column : rule (1 or 2)
- 3rd to 7th column : Go, Lick, Reward, Correct, stim

eg_Data{n,4} = time corresponding to each sample (in s) 
eg_Data{n,5} = Trial to reversal
eg_Data{n,6} = recorded session id
eg_Data{n,7} = rewarded time (0 if no reward)



