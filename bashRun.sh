python banckmark.py --device cpu --name lastTestR --config ./config/costuomConf.yaml --dataset /Users/beppe2hd/Data/Microplastiche/HMPD-Gen/images --gt /Users/beppe2hd/Data/Microplastiche/HMPD-Gen/gt.csv
python banckmark.py --device cpu --name lastTestP --config ./config/costuomConf_P.yaml --dataset /Users/beppe2hd/Data/Microplastiche/HMPD-Gen/images --gt /Users/beppe2hd/Data/Microplastiche/HMPD-Gen/gt.csv
python banckmark.py --device cpu --name lastTestA --config ./config/costuomConf_A.yaml --dataset /Users/beppe2hd/Data/Microplastiche/HMPD-Gen/images --gt /Users/beppe2hd/Data/Microplastiche/HMPD-Gen/gt.csv
python banckmark.py --device cpu --name lastTestR_Poss --config ./config/costuomConf.yaml --dataset /Users/beppe2hd/Data/Microplastiche/HMPD-Gen/images --gt /Users/beppe2hd/Data/Microplastiche/HMPD-Gen/gtPossible.csv
python banckmark.py --device cpu --name lastTestP_Poss --config ./config/costuomConf_P.yaml --dataset /Users/beppe2hd/Data/Microplastiche/HMPD-Gen/images --gt /Users/beppe2hd/Data/Microplastiche/HMPD-Gen/gtPossible.csv
python banckmark.py --device cpu --name lastTestA_Poss --config ./config/costuomConf_A.yaml --dataset /Users/beppe2hd/Data/Microplastiche/HMPD-Gen/images --gt /Users/beppe2hd/Data/Microplastiche/HMPD-Gen/gtPossible.csv


python banckmark_staticFolds.py --device cpu --name lastTestR --config ./config/costuomConf.yaml --dataset /Users/beppe2hd/Data/Microplastiche/HMPD-Gen/images
