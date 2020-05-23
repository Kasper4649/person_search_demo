# -*- coding:utf-8 -*-
import argparse
import logging
from concurrent import futures
import torch
import grpc
import io
import yaml

from message import message_pb2
from message import message_pb2_grpc
from search import detect


class Search(message_pb2_grpc.SearchServiceServicer):

    def Search(self, request, context):
        data_stream = request.file
        data = io.BytesIO(data_stream)
        with open('data/samples/' + request.name, 'wb') as f:
            f.write(data.read())

        parser = argparse.ArgumentParser()
        parser.add_argument('--cfg', type=str, default='cfg/yolov3.cfg', help="模型配置文件路径")
        parser.add_argument('--data', type=str, default='data/coco.data', help="数据集配置文件所在路径")
        parser.add_argument('--weights', type=str, default='weights/yolov3.weights', help='模型权重文件路径')
        parser.add_argument('--images', type=str, default='data/samples', help='需要进行检测的图片文件夹')
        parser.add_argument('-q', '--query', default=r'query', help='查询图片的读取路径')
        parser.add_argument('--img-size', type=int, default=416, help='输入分辨率大小')
        parser.add_argument('--conf-thres', type=float, default=0.1, help='物体置信度阈值')
        parser.add_argument('--nms-thres', type=float, default=0.4, help='NMS 阈值')
        parser.add_argument('--dist_thres', type=float, default=1.4, help='行人图片距离阈值，小于这个距离，就认为是该行人')
        parser.add_argument('--fourcc', type=str, default='vp80',
                            help='fourcc output video codec (verify ffmpeg support)')
        parser.add_argument('--output', type=str, default='output', help='检测后的图片或视频保存的路径')
        parser.add_argument('--half', default=False, help='是否采用半精度 FP16 进行推理')
        parser.add_argument('--webcam', default=False, help='是否使用摄像头进行检测')
        opt = parser.parse_args()
        print(opt)

        with torch.no_grad():
            detect(opt)

        with open("config.yaml", "r") as f:
            config = yaml.load(f)
        output_url = config["OUTPUT_BASE_URL"] + request.name

        return message_pb2.Response(url=output_url)


def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10), options=[
        ('grpc.max_send_message_length', 60 * 1024 * 1024),
        ('grpc.max_receive_message_length', 60 * 1024 * 1024)
    ])
    message_pb2_grpc.add_SearchServiceServicer_to_server(Search(), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    server.wait_for_termination()


if __name__ == '__main__':
    logging.basicConfig()
    serve()
