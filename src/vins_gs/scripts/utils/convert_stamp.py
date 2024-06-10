

def stamp2seconds(msg):
    # 访问时间戳
    timestamp = msg.header.stamp
    
    # 转换为秒
    return timestamp.secs + timestamp.nsecs * 1e-9