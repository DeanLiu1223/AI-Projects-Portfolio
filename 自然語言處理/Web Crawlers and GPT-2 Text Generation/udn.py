# -*- coding: utf-8 -*-
import scrapy
from scrapy.spiders import CrawlSpider, Rule
from scrapy.linkextractors import LinkExtractor
from urllib.parse import quote 

class UdnKeywordSpider(CrawlSpider):
    name = 'udn_keyword' 
    custom_settings = {
        'DOWNLOAD_DELAY': 3,
        'FEED_EXPORT_ENCODING': 'utf-8',
        'LOG_LEVEL': 'INFO',
    }
    allowed_domains = ['udn.com']

    rules = [
        Rule(LinkExtractor(allow=r'/news/story/\d+/\d+'), callback='parse_item', follow=False)
    ]

    def __init__(self, keyword=None, *args, **kwargs):
        """
        scrapy crawl udn_keyword -a keyword="你的關鍵字"
        """
        super(UdnKeywordSpider, self).__init__(*args, **kwargs) 
        if keyword is None:
            raise ValueError("請使用 -a keyword='您的關鍵字' 來提供搜尋關鍵字")
        self.keyword = keyword
        self.logger.info(f"收到的搜尋關鍵字: {self.keyword}")

    def _build_search_url(self):
        """
        根據關鍵字建立 UDN 搜尋 URL。
        """
        encoded_keyword = quote(self.keyword)
        return f'https://udn.com/search/word/2/{encoded_keyword}'

    def start_requests(self):
        """
        產生第一個要爬取的請求 (搜尋結果頁)。
        """
        search_url = self._build_search_url()
        self.logger.info(f"開始從搜尋頁面爬取: {search_url}")
        yield scrapy.Request(search_url, callback=None)

    def parse_item(self, response):
        """
        解析最終的新聞文章頁面。
        """
        self.logger.info(f'正在解析文章: {response.url}')
        title = response.css('h1.article-content__title::text').get()
        ps = response.css('div.article-content__paragraph p::text').getall()
        content = ''.join(ps).strip()
        url = response.url

        if title and content:
            yield {
                #'keyword': self.keyword, # 將搜尋關鍵字也加入結果中
                'title': title.strip(),
                'content': content,
                #'url': url,
            }
        else:
            self.logger.warning(f'無法從此頁面提取標題或內容: {response.url}')