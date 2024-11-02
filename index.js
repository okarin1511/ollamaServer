import express from 'express';
import { createProxyMiddleware } from 'http-proxy-middleware';

const app = express();
const PORT = 8080;
const API_KEY = '9344c017e52349577126ef0a9565a61adacc9ee5d3281b0546f48a2de4b5e7df';
const OLLAMA_URL = 'http://localhost:11434';


function validateApiKey(req, res, next) {
    const authHeader = req.headers['authorization'];
    if (authHeader === `Bearer ${API_KEY}`) {
        next();
    } else {
        res.status(401).send('Unauthorized');
    }
}

function logRequestBody(req, res, next) {
    let body = '';
    req.on('data', (chunk) => {
        body += chunk;
    });
    req.on('end', () => {
        console.log(`Request Body: ${body}`);
        req.body = body;
        next();
    });
}

const proxyOptions = {
    target: OLLAMA_URL,
    changeOrigin: true,
    onProxyReq: (proxyReq, req, res) => {
        proxyReq.setHeader('X-Forwarded-Host', req.headers['host']);
        proxyReq.setHeader('Host', new URL(OLLAMA_URL).host);
        if (req.headers.accept === 'text/event-stream') {
            res.setHeader('Content-Type', 'text/event-stream');
            res.setHeader('Cache-Control', 'no-cache');
            res.setHeader('Connection', 'keep-alive');
        }
    },
    onProxyRes: (proxyRes, req, res) => {
        if (req.headers.accept === 'text/event-stream') {
            proxyRes.headers['Content-Type'] = 'text/event-stream';
        }
    },
};

app.use('/v1', validateApiKey, logRequestBody, createProxyMiddleware(proxyOptions));

app.listen(PORT, () => {
    console.log(`Server is running on http://0.0.0.0:${PORT}`);
});
