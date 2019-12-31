.PHONY: data clean install
data: 
	wget --progress=bar -nc -i data_listing -P data
	gzip -d data/*

install: 
	pip install -r requirements.txt

clean:
	rm data/*
