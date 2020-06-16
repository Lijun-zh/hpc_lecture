Final report  20M52038

I implemented the parallelization for solving the NS equations of the cavity flow.

'1_cavity.py' is the python program that summarizes the job in the notebook '10_cavity.ipynb', with the same boundary condition and initial condition settings. It stops at a certain convergence point, which is specified by variable 'eps' in the program. Running it will print the number of steps for convergence, the elapsed time and sums of the absolute values of u, v, p. It will also generate 'python_result.png' as the plot of the solution.

'2_ported.cpp' is the ported version of the program in c++, with all the same conditions. It also stops at a specified convergence point and prints the number of steps and sums of absolute u, v, p for a sanity check.

'3_cuda.cu' is the Cuda version of the program. Note that since some Cuda functions such as atomAdd cannot operate on double data, variables are defined as float instead of double, which lead to a slight difference in the outputs compared with the results that we gained before. However, the calculating speed is faster. This program also outputs a file storing the data, which can be read and plotted by the python program '4_cuda_plot.py'.

Here are the results on my local machine which has an E3-1230v5 and a GTX 1660S. Also, the generated plots are attached in the directory.

$ python3 1_cavity.py
Reaching convergence after 6605 steps.
Elapsed time is 8.621 s.
Sum(|u|)= 219.63547247647466
Sum(|v|)= 129.10805866433103
Sum(|p|)= 175.6803115004912
Plot is saved as python_result.png
$ g++ 2_ported.cpp; ./a.out
Reaching convergence after 6605 steps.
Elapsed time is 20.924257 s.
Sum(|u|)=219.635472
Sum(|v|)=129.108059
Sum(|p|)=175.680312
$ nvcc 3_cuda.cu; ./a.out
Reaching convergence after 6607 steps.
Elapsed time is 4.115154 s.
Sum(|u|)=217.635254
Sum(|v|)=129.107635
Sum(|p|)=175.680054
$ python3 4.cuda_plot.py
Plot is saved as cuda_result.png
