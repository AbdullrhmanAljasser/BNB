import numpy as np
import pandas as pd
import sklearn as sk
from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import csr_matrix
class BNB():

    _cVec = None
    _features = None
    _targets = None
    _TS = None
    _M = None
    _indicators = None
    _theta_one = None
    _theta_zero = None
    _theta_j_one = []
    _theta_j_zero = []
    
    def __init__(self,features,targets):
        #Expecting csr_matrix
        ##Expecting features and targets (Targets are single dimension array)
        self._cVec = CountVectorizer(binary=True ,min_df=0.01) ##TODO ADD min_df, max_df,etc...
        self._targets = targets
        self._TS = targets.shape[0]
        self._numO = (self._targets == 1).sum()
        self._numZ = (self._targets == 0).sum()
        self._theta_one = (self._targets == 1).sum()/targets.shape[0]
        self._theta_zero = (self._targets == 0).sum()/targets.shape[0]
        self._indicators = self._cVec.fit_transform(features)
        self._features = self._cVec.get_feature_names()
        self._M = len(self._features)
        print(self._M)
        ##TODO: Try looping over the indicators individually and from there use the self._target to check if suck value shall be
        #updated in the corrosponding theta
        for j in range(self._M):
            self._theta_j_one.append((self._indicators.getcol(j).toarray()[self._targets.to_numpy() == 1].sum()+1)/(self._numO+2))
            self._theta_j_zero.append((self._indicators.getcol(j).toarray()[self._targets.to_numpy() == 0].sum()+1)/(self._numZ+2))
        
        
    def predict(self,features):
        predL = []
        
        features = self._cVec.transform(features).toarray()
        
        for i in range(features.shape[0]):
            tot = np.log(np.array(self._theta_j_one)/np.array(self._theta_j_zero))
            totmt= (1-np.array(self._theta_j_one))
            totmb=(1-np.array(self._theta_j_zero))
            totm= totmt/totmb
            totm=np.log(totm)
            leftS = features[i]@tot
            RighS = (1-features[i]  )@totm
            logL = leftS + RighS
            logL = logL + np.log(self._theta_one/self._theta_zero)
            predL.append(logL)
        
        x = np.sign(np.array(predL))
        x[x<0] = 0
        return x
        
    def eval_acc(self,features,targets):
        pred = self.predict(features)
        return np.mean(pred == targets)
    
    
##Creating BNB
data = pd.read_csv('train.csv', sep=',',header=0)
data = data.replace('negative',0).replace('positive',1)
feat = data['review']
targ = data['sentiment']

test = BNB(feat,targ)
print(test.predict(["Where to start? Some guy has some Indian pot that he's cleaning, and suddenly Skeletor attacks. He hits a woman in the neck with an axe, she falls down, but then gets up and is apparently uninjured. She runs into the woods, and it turns out there's the basement of a shopping center out there in the woods. She meets a utility worker and Skeletor attacks again. Luckily, like any good utility worker, he's got a gun and shoots at the guy. Doesn't work, everything starts on fire.Cut to some people walking through the woods. Even though they've been hiking together for some time, they sit down and introduce themselves to each other. Wouldn't they have probably done that when they first met? Anyhow, they're  Delta team members (undercover, I suppose, because that way they don't have to pay to dress them in uniforms). The cute girls are various things such as a sniper school instructor and, oh, I can't remember the rest. It doesn't matter. Eventually they all take their guns out and immediately start aiming them at various things. ? Anyhow, they meet an","I would not have known about this film if not for its ""surprise"" Oscar nomination for Best Animated Feature film. Thankfully, it came to pass that I was able to watch this animated little treasure.<br /><br />The story is about the child Brendan who was the nephew of the imposing and overprotective Abbot of the township of Kells. The main pre-occupation of the Abbot is to build a wall to protect Kells from the attacking Vikings. One day, Aiden, the renowned illustrator from Iona, sought refuge with them. Aiden opens Brendan's eyes to the art of illustration and the lure of the outside world. Along the way, Brendan befriended the white forest sprite Aisling, as he sought to recover an ancient crystal invaluable to the meticulous art of book illustration.<br /><br />""The Secret of Kells"" is unlike most of the animation released these days. It is a throwback of sorts as the illustrations are done in stark geometric lines and design without much care for realism, as much as symbolism. The movements of these lines are reminiscent of the simplistic yet fluid animation style used at the beginning sequence of ""Kung Fu Panda."" However, it is the magnificent use of color that is the main source of wonderment for the audience. The reds used in the Viking invasion sequence is unforgettably haunting.<br /><br />Try to catch this quiet gem of a film. It is a welcome respite from all the senseless bombast of current animated fare such as ""Monsters vs. Aliens"" and the like. The sparse Celtic musical score is effective in evoking the sense of fantasy that imbues the film. OK, the story might be a little shallow and the ending a bit wanting. I would have liked to know more about the Book that Brendan and Aiden was working on. But the clear star of this film is clearly its amazing stylized artwork, said to be based on the artwork in the real Book of Kells.","Dull, predictable and uninteresting story of a man contaminated by a chemical substance (Weller) who goes on across the country just to find his ex-wife and children; meanwhile, he kills everyone in his way only by a single touch of his hands. In his dangerous track, a doctor (Hurt) and a young reporter (Natasha) try to stop the man. The movie has a not original premise but even though could be much better. The final result is just a movie without suspense or gritting moments. Even the good cast is completely waste. I give this a 4 (four)."]))