# ram
"The son of Hezron is Ram." Ram allows to experiment with different deep Q learning algorithms to express feelings when he is talking.

## Genealogy of Ram

1. [Samu](https://github.com/nbatfai/samu)
The main purpose of this project is to allow the evaluation and verification of the results of the paper entitled **"A disembodied developmental robotic agent called Samu Bátfai"**. It is our hope that Samu will be the ancestor of developmental robotics chatter bots that will be able to chat in natural language like humans do.
2. [Isaac](https://github.com/nbatfai/isaac)
*"The son of Samu is Isaac."* The project called Isaac is a case study of using deep Q learning with neural networks for predicting the next sentence of a conversation.
3. [Jacob](https://github.com/nbatfai/jacob)
*"The son of Isaac is Jacob."* The project called Jacob is an experiment to replace Isaac's (GUI based) visual imagination with a character console. 
4. [Judah](https://github.com/nbatfai/judah)
*"The son of Jacob is Judah."* In the project called Judah we equip Jacob with a text-based user interface.
5. [Super-Judah](https://github.com/nbatfai/super-judah)
6. [Perez](https://github.com/nbatfai/perez)
*"The son of Judah is Perez."* The project called Perez allows to perform experiments to test several parameter settings. Here each experiment is implemented as a separate git branch.
7. [Hezron](https://github.com/nbatfai/hezron)
*"The son of Perez is Hezron."* The project called Hezron gives the opportunity to experiment with different arrangements of SPO triplets in the visual imagery.
8. [Ram](https://github.com/nbatfai/ram)
*"The son of Hezron is Ram."* The project called Ram allows to experiment with different deep Q learning algorithms to express feelings when he is talking.


### Installation of Ram

Building Samu (Ram), uncomment the appropriate lines in the file CMakeLists.txt. For example, if you have CUDA installed on your system Samu may be built with CUDA support, 
```
set(CUDA_LAYERS ON)
#set(CUDA_LAYERS OFF)
```
Or, if you want to use TUI you need uncomment the following line:
```
set(DISP_CURSES ON)
#set(DISP_CURSES OFF)
```
After finishing the edit of the CMakeLists.txt type `cmake .`
```
nbatfai@orchmach:~/GitHubRepos/judah$ cmake .
-- Found CUDA: /usr/local/cuda-7.0 (found version "7.0") 
-- Found Curses: /usr/lib/x86_64-linux-gnu/libcurses.so  
-- Try OpenMP C flag = [-fopenmp]
-- Performing Test OpenMP_FLAG_DETECTED
-- Performing Test OpenMP_FLAG_DETECTED - Success
-- Try OpenMP CXX flag = [-fopenmp]
-- Performing Test OpenMP_FLAG_DETECTED
-- Performing Test OpenMP_FLAG_DETECTED - Success
-- Found OpenMP: -fopenmp  
-- Boost version: 1.55.0
-- Found the following Boost libraries:
--   date_time
-- Found ZLIB: /usr/lib/x86_64-linux-gnu/libz.so (found version "1.2.8") 
-- Found PNG: /usr/lib/x86_64-linux-gnu/libpng.so (found suitable version "1.2.51", minimum required is "1.2.9") 
-- Found Freetype: /usr/lib/x86_64-linux-gnu/libfreetype.so (found version "2.5.2") 
-- Found PNGwriter: /home/nbatfai/NLP/pngwriter/lib/libpngwriter.so;/usr/lib/x86_64-linux-gnu/libpng.so;/usr/lib/x86_64-linux-gnu/libz.so;/usr/lib/x86_64-linux-gnu/libfreetype.so (found version "0.5.4") 
-- Link Grammar  found.
-- Configuring done
-- Generating done
-- Build files have been written to: /home/nbatfai/GitHubRepos/judah
```
If `cmake .` finishes without problems you can run the `make` command to build Samu.

### Running

To run Samu (Ram), type the following command
```
./samu 2>out
```
If you use ncurses you will see something like this:

![hezrontui2](https://cloud.githubusercontent.com/assets/3148120/10677104/b74066f0-790a-11e5-9044-ead867835a7f.png)

In another terminal window, run the command
```
tail -f out|grep iter
```
to see Samu's "training curve".

See the project's wiki page for further information. 

# Samu
The purpose of this project is only to allow the evaluation and verification of the results of the paper entitled **"A disembodied developmental robotic agent called Samu Bátfai"**. This paper presents Q learning with neural networks approximators used by Samu. 

It is our hope that Samu will be the ancestor of developmental robotics chatter bots that will be able to chat in natural language like humans do.


