"""
Sentiment analysis for cryptocurrency market using news and social media data
"""
import asyncio
import aiohttp
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import json
import re
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import structlog

from config.settings import settings

logger = structlog.get_logger(__name__)

class SentimentAnalyzer:
    """Analyze market sentiment from news and social media"""
    
    def __init__(self):
        self.vader_analyzer = SentimentIntensityAnalyzer()
        self.news_api_key = settings.news_api_key
        self.twitter_bearer_token = settings.twitter_bearer_token
        
    async def analyze_sentiment(self, pair_id: str) -> Dict[str, Any]:
        """Analyze sentiment for a specific trading pair"""
        try:
            logger.info("Starting sentiment analysis", pair_id=pair_id)
            
            # Extract crypto symbol from pair_id (e.g., btc_idr -> btc)
            crypto_symbol = pair_id.split('_')[0].upper()
            
            # Gather sentiment data from multiple sources
            news_sentiment = await self._analyze_news_sentiment(crypto_symbol)
            social_sentiment = await self._analyze_social_sentiment(crypto_symbol)
            
            # Combine sentiments
            combined_sentiment = self._combine_sentiments(news_sentiment, social_sentiment)
            
            logger.info("Sentiment analysis completed", 
                       pair_id=pair_id,
                       news_score=news_sentiment.get('score', 0),
                       social_score=social_sentiment.get('score', 0),
                       combined_score=combined_sentiment.get('score', 0))
            
            return combined_sentiment
            
        except Exception as e:
            logger.error("Failed to analyze sentiment", pair_id=pair_id, error=str(e))
            return {"score": 0.0, "label": "neutral", "confidence": 0.0}
    
    async def _analyze_news_sentiment(self, crypto_symbol: str) -> Dict[str, Any]:
        """Analyze sentiment from cryptocurrency news"""
        try:
            if not self.news_api_key:
                logger.warning("News API key not configured, skipping news sentiment")
                return {"score": 0.0, "label": "neutral", "confidence": 0.0}
            
            # Search for recent news about the cryptocurrency
            news_articles = await self._fetch_news_articles(crypto_symbol)
            
            if not news_articles:
                return {"score": 0.0, "label": "neutral", "confidence": 0.0}
            
            # Analyze sentiment of each article
            sentiments = []
            for article in news_articles:
                title = article.get('title', '')
                description = article.get('description', '')
                content = f"{title}. {description}"
                
                if content.strip():
                    sentiment = self._analyze_text_sentiment(content)
                    sentiments.append(sentiment)
            
            if not sentiments:
                return {"score": 0.0, "label": "neutral", "confidence": 0.0}
            
            # Calculate average sentiment
            avg_score = sum(s['score'] for s in sentiments) / len(sentiments)
            avg_confidence = sum(s['confidence'] for s in sentiments) / len(sentiments)
            
            # Determine label
            if avg_score > 0.1:
                label = "positive"
            elif avg_score < -0.1:
                label = "negative"
            else:
                label = "neutral"
            
            return {
                "score": avg_score,
                "label": label,
                "confidence": avg_confidence,
                "source": "news",
                "article_count": len(sentiments)
            }
            
        except Exception as e:
            logger.error("Failed to analyze news sentiment", crypto_symbol=crypto_symbol, error=str(e))
            return {"score": 0.0, "label": "neutral", "confidence": 0.0}
    
    async def _fetch_news_articles(self, crypto_symbol: str) -> List[Dict[str, Any]]:
        """Fetch recent news articles about cryptocurrency"""
        try:
            # Calculate date range (last 7 days)
            to_date = datetime.now()
            from_date = to_date - timedelta(days=7)
            
            # Prepare search query
            query = f"{crypto_symbol} OR {crypto_symbol.lower()} OR {self._get_crypto_name(crypto_symbol)}"
            
            # API endpoint
            url = "https://newsapi.org/v2/everything"
            params = {
                "q": query,
                "apiKey": self.news_api_key,
                "language": "en",
                "sortBy": "publishedAt",
                "from": from_date.strftime("%Y-%m-%d"),
                "to": to_date.strftime("%Y-%m-%d"),
                "pageSize": 50
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data.get('articles', [])
                    else:
                        logger.warning("News API request failed", status=response.status)
                        return []
                        
        except Exception as e:
            logger.error("Failed to fetch news articles", crypto_symbol=crypto_symbol, error=str(e))
            return []
    
    async def _analyze_social_sentiment(self, crypto_symbol: str) -> Dict[str, Any]:
        """Analyze sentiment from social media (Twitter/Reddit)"""
        try:
            # For now, implement a basic social sentiment analysis
            # This would be expanded to use Twitter API v2 and Reddit API
            
            # Mock social sentiment data
            social_sentiments = []
            
            # Twitter sentiment (placeholder)
            twitter_sentiment = await self._analyze_twitter_sentiment(crypto_symbol)
            if twitter_sentiment:
                social_sentiments.append(twitter_sentiment)
            
            # Reddit sentiment (placeholder)
            reddit_sentiment = await self._analyze_reddit_sentiment(crypto_symbol)
            if reddit_sentiment:
                social_sentiments.append(reddit_sentiment)
            
            if not social_sentiments:
                return {"score": 0.0, "label": "neutral", "confidence": 0.0}
            
            # Calculate average sentiment
            avg_score = sum(s['score'] for s in social_sentiments) / len(social_sentiments)
            avg_confidence = sum(s['confidence'] for s in social_sentiments) / len(social_sentiments)
            
            # Determine label
            if avg_score > 0.1:
                label = "positive"
            elif avg_score < -0.1:
                label = "negative"
            else:
                label = "neutral"
            
            return {
                "score": avg_score,
                "label": label,
                "confidence": avg_confidence,
                "source": "social",
                "platform_count": len(social_sentiments)
            }
            
        except Exception as e:
            logger.error("Failed to analyze social sentiment", crypto_symbol=crypto_symbol, error=str(e))
            return {"score": 0.0, "label": "neutral", "confidence": 0.0}
    
    async def _analyze_twitter_sentiment(self, crypto_symbol: str) -> Optional[Dict[str, Any]]:
        """Analyze Twitter sentiment (placeholder implementation)"""
        try:
            # This would use Twitter API v2 to fetch recent tweets
            # For now, return a placeholder
            
            # Mock tweets analysis
            mock_tweets = [
                f"{crypto_symbol} is looking bullish today! 🚀",
                f"Not sure about {crypto_symbol} right now, market seems uncertain",
                f"{crypto_symbol} to the moon! Great investment opportunity",
                f"Selling my {crypto_symbol} position, too risky",
                f"{crypto_symbol} price action is interesting, waiting for confirmation"
            ]
            
            sentiments = []
            for tweet in mock_tweets:
                sentiment = self._analyze_text_sentiment(tweet)
                sentiments.append(sentiment)
            
            if not sentiments:
                return None
            
            avg_score = sum(s['score'] for s in sentiments) / len(sentiments)
            avg_confidence = sum(s['confidence'] for s in sentiments) / len(sentiments)
            
            return {
                "score": avg_score,
                "confidence": avg_confidence,
                "platform": "twitter",
                "post_count": len(sentiments)
            }
            
        except Exception as e:
            logger.error("Failed to analyze Twitter sentiment", crypto_symbol=crypto_symbol, error=str(e))
            return None
    
    async def _analyze_reddit_sentiment(self, crypto_symbol: str) -> Optional[Dict[str, Any]]:
        """Analyze Reddit sentiment (placeholder implementation)"""
        try:
            # This would use Reddit API to fetch posts from crypto subreddits
            # For now, return a placeholder
            
            # Mock Reddit posts analysis
            mock_posts = [
                f"DD: Why {crypto_symbol} is undervalued and could 10x",
                f"{crypto_symbol} technical analysis - bearish pattern forming",
                f"Just bought more {crypto_symbol}, DCA strategy",
                f"Warning: {crypto_symbol} whale movements detected",
                f"{crypto_symbol} partnership announcement coming soon"
            ]
            
            sentiments = []
            for post in mock_posts:
                sentiment = self._analyze_text_sentiment(post)
                sentiments.append(sentiment)
            
            if not sentiments:
                return None
            
            avg_score = sum(s['score'] for s in sentiments) / len(sentiments)
            avg_confidence = sum(s['confidence'] for s in sentiments) / len(sentiments)
            
            return {
                "score": avg_score,
                "confidence": avg_confidence,
                "platform": "reddit",
                "post_count": len(sentiments)
            }
            
        except Exception as e:
            logger.error("Failed to analyze Reddit sentiment", crypto_symbol=crypto_symbol, error=str(e))
            return None
    
    def _analyze_text_sentiment(self, text: str) -> Dict[str, Any]:
        """Analyze sentiment of a single text using multiple methods"""
        try:
            # Clean text
            cleaned_text = self._clean_text(text)
            
            if not cleaned_text:
                return {"score": 0.0, "confidence": 0.0}
            
            # VADER sentiment analysis
            vader_scores = self.vader_analyzer.polarity_scores(cleaned_text)
            vader_compound = vader_scores['compound']
            
            # TextBlob sentiment analysis
            blob = TextBlob(cleaned_text)
            textblob_polarity = blob.sentiment.polarity
            
            # Combine scores (weighted average)
            combined_score = (vader_compound * 0.6) + (textblob_polarity * 0.4)
            
            # Calculate confidence based on agreement between methods
            agreement = 1 - abs(vader_compound - textblob_polarity)
            confidence = min(agreement, abs(combined_score))
            
            return {
                "score": combined_score,
                "confidence": confidence,
                "vader_score": vader_compound,
                "textblob_score": textblob_polarity
            }
            
        except Exception as e:
            logger.error("Failed to analyze text sentiment", error=str(e))
            return {"score": 0.0, "confidence": 0.0}
    
    def _clean_text(self, text: str) -> str:
        """Clean text for sentiment analysis"""
        try:
            # Remove URLs
            text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
            
            # Remove special characters but keep emoticons
            text = re.sub(r'[^\w\s\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF]', ' ', text)
            
            # Remove extra whitespace
            text = re.sub(r'\s+', ' ', text).strip()
            
            return text
            
        except Exception as e:
            logger.error("Failed to clean text", error=str(e))
            return text
    
    def _combine_sentiments(self, news_sentiment: Dict[str, Any], social_sentiment: Dict[str, Any]) -> Dict[str, Any]:
        """Combine news and social sentiment scores"""
        try:
            news_score = news_sentiment.get('score', 0.0)
            news_confidence = news_sentiment.get('confidence', 0.0)
            
            social_score = social_sentiment.get('score', 0.0)
            social_confidence = social_sentiment.get('confidence', 0.0)
            
            # Weighted average based on confidence
            total_confidence = news_confidence + social_confidence
            
            if total_confidence > 0:
                combined_score = (
                    (news_score * news_confidence) + 
                    (social_score * social_confidence)
                ) / total_confidence
                
                combined_confidence = total_confidence / 2
            else:
                combined_score = 0.0
                combined_confidence = 0.0
            
            # Determine label
            if combined_score > 0.1:
                label = "positive"
            elif combined_score < -0.1:
                label = "negative"
            else:
                label = "neutral"
            
            return {
                "score": combined_score,
                "label": label,
                "confidence": combined_confidence,
                "news_sentiment": news_sentiment,
                "social_sentiment": social_sentiment,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error("Failed to combine sentiments", error=str(e))
            return {"score": 0.0, "label": "neutral", "confidence": 0.0}
    
    def _get_crypto_name(self, symbol: str) -> str:
        """Get full cryptocurrency name from symbol"""
        crypto_names = {
            'BTC': 'Bitcoin',
            'ETH': 'Ethereum',
            'BNB': 'Binance Coin',
            'ADA': 'Cardano',
            'SOL': 'Solana',
            'DOT': 'Polkadot',
            'LINK': 'Chainlink',
            'UNI': 'Uniswap',
            'LTC': 'Litecoin',
            'XRP': 'Ripple'
        }
        
        return crypto_names.get(symbol.upper(), symbol)
