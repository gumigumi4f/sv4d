for model_dir in `\find ../models/ -maxdepth 1 -mindepth 1 -type d | sort`
do
	echo "Evaluate model...  Model: ${model_dir}"
	python evaluate_wsd.py ${model_dir} ../corpus/ALL.data.xml ../corpus/ALL.gold.key.txt ../corpus
done
