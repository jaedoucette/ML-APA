.PHONY: data clean
data: 
	wget --progress=bar -nc -i data_listing -P data
	gzip -d data/*

clean:
	rm data/*
