for beta_dict in `seq 0.25 0.1 0.45`
do
	for beta_reward in `seq 0.25 0.1 0.45`
	do
		echo "Training model: bd${beta_dict}_br${beta_reward}  "
		mkdir "../models/bd${beta_dict}_br${beta_reward}"
		../bin/sv4d training -training_corpus ../corpus/WestburyLab.Wikipedia.ProcessedCorpus.txt -synset_data_file ../corpus/sense.txt -model_dir ../models/bd${beta_dict}_br${beta_reward} -thread_num 32 -epochs 10 -min_beta_dict ${beta_dict} -min_beta_reward ${beta_reward}
	done
done
