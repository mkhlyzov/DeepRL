import time


# It does not include time elapsed during sleep
# def GET_TIME(): return time.process_time()

# It does include time elapsed during sleep
def GET_TIME(): return time.perf_counter()
