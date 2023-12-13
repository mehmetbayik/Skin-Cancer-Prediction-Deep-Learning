"data" folder must be in same folder with q1main.ipynb and q2main.ipynb file.

For question 1, run q1main.ipynb file. It should run without any problem.

For question 2, I used q2main.ipynb file. But that file is very complicated since I tried to do everything in one file. Also, my code saves all the results in model-comparison folder with using current date-time to compare models easily. Since I use mac, I don't know if it will create any problem in windows or colab. Since I used '/' in file paths, it may create problem in windows. I am not sure.

In order to grader can test my Q2 code easily, I created a new file called q2_reproducible.ipynb. It is a simplified version of q2main.ipynb file. It does not save any results. It just prints the results for default and best parameters. You can run this file to check if my code is working or not.

For Q2.2 Plots of 13 different models, I used q2_plot.ipynb file. It is a simplified version of q2main.ipynb file for just plots. But it plots the history from my saved results. In order to run this file, I added my model-comparison folder to zip file. You can run this file to check if my code is working or not. 

This should be all for Q2. I also added my best-models.txt file to zip file. It contains the parameters for each model. 

I also added my model-comparison folder to zip file. It contains all the results for each model.


Used dependencies are:
  - python=3.8.17
  - numpy=1.24.0
  - matplotlib=3.7.1
  - pandas=2.0.2 


