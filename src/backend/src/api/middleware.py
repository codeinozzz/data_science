from fastapi import Request, HTTPException
from collections import defaultdict
import time


class SimpleRateLimiter:
    def __init__(self, requests_per_minute=10):
        self.requests_per_minute = requests_per_minute
        self.requests = defaultdict(list)
    
    def is_allowed(self, client_ip: str) -> bool:
        now = time.time()
        minute_ago = now - 60
        
        self.requests[client_ip] = [
            req_time for req_time in self.requests[client_ip] 
            if req_time > minute_ago
        ]
        
        if len(self.requests[client_ip]) >= self.requests_per_minute:
            return False
        
        self.requests[client_ip].append(now)
        return True


rate_limiter = SimpleRateLimiter(requests_per_minute=10)


async def rate_limit_middleware(request: Request, call_next):
    client_ip = request.client.host
    
    if request.url.path.startswith("/search"):
        if not rate_limiter.is_allowed(client_ip):
            raise HTTPException(429, "Too many requests. Please try again later.")
    
    response = await call_next(request)
    return response