FROM modelflow:8081/component/base_images:v0.0.2
ADD . .
RUN pip install -i https://pypi.douban.com/simple/ -r requirements.txt
