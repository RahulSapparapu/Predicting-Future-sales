ML PROJECT README
team 7
predicting future sales

our main model is xg_boost,its pickle file is off the size 55,177 kb
as my pickle is small we are submitting in the zip file

command to compile our code 

python predict.py train.py test.py

our pickel file
filename = 'model_xgboost.pkl'
outfile = open(filename,'wb')
pickle.dump(model,outfile)
outfile.close()



DRIVE LINK FOR PICKLE FILE:
https://drive.google.com/open?id=1ISNvDgRA776OMnvFSehUZvN2zhFB5rtK