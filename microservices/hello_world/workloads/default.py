from locust import task, between, constant_pacing
from locust.contrib.fasthttp import FastHttpUser

class MyUser(FastHttpUser):
    network_timeout = 2
    wait_time = constant_pacing(2)

    @task
    def index(self):
        response = self.client.get("")
