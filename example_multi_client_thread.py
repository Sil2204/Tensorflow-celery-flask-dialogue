import argparse
import requests
import random
from threading import Thread
import time
import sys

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     description="tensorflow + celery + flask client thread")

parser.add_argument('--file', default='/home/sil2204/sentence_test',
                        help='directory where sentence file')

args = parser.parse_args()

class ClientThread(Thread):
  def __init__(self, sentence): 
    Thread.__init__(self) 
    self.sentence = sentence
 
  def run(self): 
    p = {'sentence':self.sentence}
    r = requests.get('http://localhost:5000/test', params = p)
    #print r.json()
    gto = r.json()['goto']
    start_time = time.time()
    r2 = requests.get("http://localhost:5000/test/result/{}".format(gto))
    elapsed_time = time.time() - start_time
    print r2.json()['result'.decode('utf-8')]
    f = file("result.log", "a")
    sutf8 = u''.join(r2.json()['result']).strip()
    print sutf8
    sutf9 = sutf8.encode('UTF-8')
    f.write(sutf9 + "\t" + "Elapsed_time=" + str(elapsed_time) + "\n")
    f.close

def main():
  # Send request
  threads = [] 
  with open(args.file, 'rb') as f:
    for sentence in f.read().split('\n'):
      newthread = ClientThread(sentence) 
      newthread.start() 
      threads.append(newthread) 
 
    for t in threads: 
      t.join() 


if __name__ == '__main__':
  main()

#sentence = raw_input("translate_client Enter sentence/ Enter exit:") 

