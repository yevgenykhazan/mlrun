# Copyright 2018 Iguazio
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
import datetime
import time


def handler(context, time_to_sleep=1):
    print("started", str(datetime.datetime.now()))
    print(f"Sleeping for {time_to_sleep} seconds")
    context.log_result("started", str(datetime.datetime.now()))
    time.sleep(int(time_to_sleep))
    context.log_result("finished", str(datetime.datetime.now()))
    print("finished", str(datetime.datetime.now()))
