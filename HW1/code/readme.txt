This directory contains code and examples for turning a directory of articles into a vector of word counts.  

To run demo, execute these commands (assuming a linux-like shell):

#########################

# first, take all the articles (Economist articles from the Asian region for this demo) and put them into one file
cat asia/*.txt > econ_asia_full.txt

# then generate the vector of word counts for that file, using the appropriate binary file
# on cygwin:  

chmod u+x text2wfreq.cygwin.exe
./text2wfreq.cygwin.exe < econ_asia_full.txt > econ_asia_full.wfreq

# or on unix:
chmod u+x text2wfreq.unix
./text2wfreq.unix < econ_asia_full.txt > econ_asia_full.wfreq

# on other systems you will need to compile yourself
cd CMU-Cam_Toolkit_v2
cd src
make install
cd ../../
CMU-Cam_Toolkit_v2/bin/text2wfreq < econ_asia_full.txt > econ_asia_full.wfreq

# compare your results to those in ./results

#########################

First we concatenate all the text files in the asia/ directory and saves them as econ_asia_full.txt.  We then use the text2wfreq program from the CMU-Cambridge Statistical Language Modeling toolkit to turn that text into a vector of word counts.  Two binary versions of that program are provided pre-compiled for use on cygwin (text2wfreq.cygwin.exe) or unix (text2wfreq.unix) systems.  If you need a binary for a different system, or are interested in the source code and further documentation, please see the instructions above for compiling the code in the ./CMU-Cam_Toolkit_v2 directory or visit the project homepage: http://www.speech.cs.cmu.edu/SLM/toolkit.html.

The results/ directory contains the output files as they should appear once you are done.

Although these tools should help you get started, you will still need to think about how to use them to get the counts (and class labels) you need from all the data in a format you can use.  Of course, you can also write your own code to do the processing in a way you are comfortable with.

As always, send e-mail to the instructor for assistance.

