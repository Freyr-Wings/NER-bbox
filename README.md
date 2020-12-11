# Solving NER tasks - Bounding Box Regression

## Install Requirements
`pip install transformers`

## Important Package
```Python
from example import visualize
```

## Demo
`visualize("May 13 Arrive in London")`

                      [CLS]      |    #may     |         #13      |   #arrive     |      #in    |   #london     |  [SEP]
            LOC              0              0              0              0              0              1              0
            ORG              0              0              0              0              0              0              0
            PER              0              0              0              0              0              0              0
           MISC              0              0              0              0              0              0              0
           
 
 `visualize("Freyr and his Chinese friends implemented this demo")`

                         [CLS]     |     #frey       |    r      |     #and      |     #his    |   #chinese   |    #friends |  #implemented   |     #this    |    #demo     |     [SEP]
            LOC              0              0              0              0              0              0              0              0              0              0              0
            ORG              0              0              0              0              0              0              0              0              0              0              0
            PER              0              1              1              0              0              0              0              0              0              0              0
           MISC              0              0              0              0              0              1              0              0   
    
    
 `visualize("Former England captain Will Carling , handed the kicking duties , finished with 20 points")`
    
                       [CLS]      |   #former     |   #england      |  #captain  |     #will     |    #carl    |     ing      |       #        |      ,    |    #handed     |     #the    |    #kicking    |    #duties   |     #        |     ,   |  #finished     |     #with      |      #20  |      #points      |    [SEP]
            LOC              0              0              1              0              0              0              0              0              0              0              0              0              0              0              0              0              0              0              0              0
            ORG              0              0              0              0              0              0              0              0              0              0              0              0              0              0              0              0              0              0              0              0
            PER              0              0              0              0              1              1              1              0              0              0              0              0              0              0              0              0              0              0              0              0
           MISC              0              0              0              0              0              0              0              0              0              0              0              0              0              0              0              0              0              0              0              0
    
    
 `visualize("The report is monitored by the British Broadcasting Corporation ( BBC )")`
 
                     [CLS]      |       #the     |     #report       |   #is  |     #monitored    |    #by     |      #the   |    #british    | #broadcasting  | #corporation  |   #      |      (        |   #bbc       |       #      |        )      |    [SEP]
            LOC              0              0              0              0              0              0              0              0              0              0              0              0              0              0              0              0
            ORG              0              0              0              0              0              0              0              1              1              1              0              0              1              0              0              0
            PER              0              0              0              0              0              0              0              0              0              0              0              0              0              0              0              0
           MISC              0              0              0              0              0              0              0              0              0              0              0              0              0              0              0              0
