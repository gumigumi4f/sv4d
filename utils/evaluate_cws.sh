for model_dir in `\find ../models/ -maxdepth 1 -mindepth 1 -type d`
do
	echo "Evaluate model...  Model: ${model_dir}"
	python evaluate_cws.py ${model_dir} ../corpus/SCWS.txt
done
