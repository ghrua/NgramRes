DEST=/mnt/task_wrapper/user_output/artifacts/

# apt-get update
# apt-get install -y cpanminus libxml-parser-perl
# cd $DEST
# pip install -U git+https://github.com/pltrdy/pyrouge
# git clone https://github.com/pltrdy/files2rouge.git     
# cd files2rouge
# python setup_rouge.py
# python setup.py install

cd $DEST

# wget http://nlp.stanford.edu/software/stanford-corenlp-full-2016-10-31.zip
# unzip stanford-corenlp-full-2016-10-31.zip
export CLASSPATH=$DEST/stanford-corenlp-full-2016-10-31/stanford-corenlp-3.7.0.jar


# wget https://cfdownload.adobe.com/pub/adobe/coldfusion/java/java11/java11015/jdk-11.0.15_linux-x64_bin.tar.gz
# tar -xvzf jdk-11.0.15_linux-x64_bin.tar.gz
export JAVA_HOME=$DEST/jdk-11.0.15/
export CLASSPATH=".:${JAVA_HOME}/lib:${JRE_HOME}/lib:$CLASSPATH"
export JAVA_PATH="${JAVA_HOME}/bin:${JRE_HOME}/bin"
export PATH="${JAVA_PATH}:$PATH"
