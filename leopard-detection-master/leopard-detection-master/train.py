from darkflow.net.build import TFNet            #darkflow is supporting library to build model     

options = {'model' : 'cfg/yolotiny.cfg',      #making dictionary for inputs to feed in model
               'load' : 'bin/yolotiny.weights',
               'batch' : 2,
               'epoch' : 200,
               'gpu' : 1.0,
               'train' : True,
               'annotation' : './xml',
               'dataset' : './Images1',
               'threshold' : 0.1
              }

tfnet = TFNet(options)                            #setting up layers
tfnet.train()                                     #train