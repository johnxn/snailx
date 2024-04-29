# coding = utf-8
from typing import Any, Dict, List, Set, Tuple, Union
from dataclasses import dataclass

from futu import (
    ModifyOrderOp,
    TrdSide,
    TrdEnv,
    TrdMarket,
    KLType,
    OpenQuoteContext,
    OrderBookHandlerBase,
    OrderStatus,
    OrderType,
    RET_ERROR,
    RET_OK,
    StockQuoteHandlerBase,
    TradeDealHandlerBase,
    TradeOrderHandlerBase,
    OpenSecTradeContext,
    OpenFutureTradeContext,
    Currency,
    SecurityFirm
)


def symbol_futu_to_snail(symbol):
    return symbol.split('.')[-1]


def symbol_snail_to_futu(symbol, market='US'):
    return f'{market}.{symbol}'


@dataclass
class FutuAccountData:
    market: str
    env: TrdEnv
    total_value: float = 0
    cash: float = 0

    def __post_init__(self):
        pass

@dataclass
class FutuPosition:
    market: str
    env: TrdEnv
    symbol: str
    side: str
    volume: float
    frozen: float
    enter_price: float
    pnl: float

    def __post_init__(self):
        pass

@dataclass
class FutuTradeData:
    market: str
    env: TrdEnv
    symbol: str
    side: str
    volume: float
    enter_price: float
    trade_id: str
    order_id: str

class FutuGateway(object):
    default_setting: Dict[str, Any] = {
        "password": "911015",
        "ip": "127.0.0.1",
        "port": 11111,
        "market": "US",
        "env": TrdEnv.REAL,
    }

    def __init__(self):
        self.quote_ctx: OpenQuoteContext = None
        self.trade_ctx: Union[OpenSecTradeContext, OpenFutureTradeContext] = None

        self.host: str = ""
        self.port: int = 0
        self.market: str = ""
        self.password: str = ""
        self.env: TrdEnv = TrdEnv.REAL

        self.trades: Set = set()

        self.on_trade = None
        self.on_order = None


    def close(self) -> None:
        """关闭接口"""
        if self.quote_ctx:
            self.quote_ctx.close()

        if self.trade_ctx:
            self.trade_ctx.close()

    def write_log(self, log):
        print(log)

    def connect(self, setting=None) -> None:
        """连接交易接口"""
        if setting is None:
            setting = FutuGateway.default_setting
        self.host: str = setting["ip"]
        self.port: int = setting["port"]
        self.market: str = setting["market"]
        self.password: str = setting["password"]
        self.env: TrdEnv = setting["env"]

        self.connect_trade()

    def connect_trade(self) -> None:
        """连接交易服务端"""
        if self.market == "HK":
            self.trade_ctx = OpenSecTradeContext(filter_trdmarket=TrdMarket.HK, host=self.host, port=self.port,)
        elif self.market == "US":
            # self.trade_ctx = OpenSecTradeContext(filter_trdmarket=TrdMarket.US, host=self.host, port=self.port,)
            self.trade_ctx = OpenFutureTradeContext(host=self.host, port=self.port, is_encrypt=None,
                                   security_firm=SecurityFirm.FUTUSECURITIES)
        elif self.market == "HK_FUTURE":
            self.trade_ctx = OpenFutureTradeContext(host=self.host, port=self.port)

        class OrderHandler(TradeOrderHandlerBase):
            gateway: FutuGateway = self

            def on_recv_rsp(self, rsp_str):
                ret_code, content = super(OrderHandler, self).on_recv_rsp(
                    rsp_str
                )
                if ret_code != RET_OK:
                    return RET_ERROR, content
                self.gateway.process_order(content)
                return RET_OK, content

        class DealHandler(TradeDealHandlerBase):
            gateway: FutuGateway = self

            def on_recv_rsp(self, rsp_str):
                ret_code, content = super(DealHandler, self).on_recv_rsp(
                    rsp_str
                )
                if ret_code != RET_OK:
                    return RET_ERROR, content
                self.gateway.process_deal(content)
                return RET_OK, content

        # 交易接口解锁
        code, data = self.trade_ctx.unlock_trade(self.password)
        if code == RET_OK:
            print("unlock futu trading succeed")
        else:
            print(f'unlock futu trading failed, reason:{data}')

        # 连接交易接口
        self.trade_ctx.set_handler(OrderHandler())
        self.trade_ctx.set_handler(DealHandler())
        self.trade_ctx.start()
        print("connect futu trade_ctx succeed")

    def process_order(self, data) -> None:
        """委托信息处理推送"""
        for ix, row in data.iterrows():
            if row["order_status"] == OrderStatus.DELETED:
                continue

            # direction, offset = DIRECTION_FUTU2VT[row["trd_side"]]
            # symbol, exchange = convert_symbol_futu2vt(row["code"])
            # order: OrderData = OrderData(
            #     symbol=symbol,
            #     exchange=exchange,
            #     orderid=str(row["order_id"]),
            #     direction=direction,
            #     offset=offset,
            #     price=float(row["price"]),
            #     volume=row["qty"],
            #     traded=row["dealt_qty"],
            #     status=STATUS_FUTU2VT[row["order_status"]],
            #     datetime=generate_datetime(row["create_time"]),
            #     gateway_name=self.gateway_name,
            # )
            callable(self.on_trade) and self.on_order(row)

    def process_deal(self, data) -> None:
        """成交信息处理推送"""
        for ix, row in data.iterrows():
            tradeid: str = str(row["deal_id"])
            if tradeid in self.trades:
                continue
            self.trades.add(tradeid)

            # direction, offset = DIRECTION_FUTU2VT[row["trd_side"]]
            # symbol, exchange = convert_symbol_futu2vt(row["code"])
            # trade: TradeData = TradeData(
            #     symbol=symbol,
            #     exchange=exchange,
            #     direction=direction,
            #     offset=offset,
            #     tradeid=tradeid,
            #     orderid=row["order_id"],
            #     price=float(row["price"]),
            #     volume=row["qty"],
            #     datetime=generate_datetime(row["create_time"]),
            #     gateway_name=self.gateway_name,
            # )
            trade = FutuTradeData(
                market=self.market,
                env=self.env,
                symbol=symbol_futu_to_snail(row['code']),
                side=row["trd_side"],
                volume=row['qty'],
                enter_price=row['price'],
                trade_id=row["deal_id"],
                order_id=row["order_id"]
            )
            callable(self.on_trade) and self.on_trade(trade)

    def send_order(self, symbol, price, volume, side, order_type=OrderType.NORMAL) -> str:
        """委托下单"""
        # 设置调整价格限制
        symbol = symbol_snail_to_futu(symbol, self.market)
        if side in [TrdSide.BUY, TrdSide.BUY_BACK]:
            adjust_limit: float = 0.05
        else:
            adjust_limit: float = -0.05

        code, data = self.trade_ctx.place_order(
            price,
            volume,
            symbol,
            side,
            order_type,
            trd_env=self.env,
            adjust_limit=adjust_limit,
        )

        if code:
            print(f"send order failed：{data}")
            return ""

        order_id = ''
        for ix, row in data.iterrows():
            order_id: str = str(row["order_id"])

        # self.on_order(order)
        return order_id

    def cancel_order(self, order_id) -> None:
        """委托撤单"""
        code, data = self.trade_ctx.modify_order(
            ModifyOrderOp.CANCEL, order_id, 0, 0, trd_env=self.env
        )

        if code:
            print(f"cancel order failed：{data}")

    def query_account(self):
        """查询资金"""
        code, data = self.trade_ctx.accinfo_query(trd_env=self.env, acc_id=0)

        if code:
            print(f"query account failed：{data}")
            return

        account_data_list = []
        for ix, row in data.iterrows():
            account = FutuAccountData(
                market=self.market,
                env=self.env,
                total_value=float(row['total_assets']),
                cash=float(row['cash'])
            )
            account_data_list.append(account)
        return account_data_list

    def query_account_new(self):
        account_list = self.trade_ctx.get_acc_list()
        account_df = account_list[1]
        account_id = None
        if self.env == TrdEnv.REAL:
             account_id = account_df[account_df['trd_env'] == self.env].iloc[0]['acc_id']
        else:
            # todo
            pass

        code, data = self.trade_ctx.accinfo_query(trd_env=self.env, acc_id=account_id,refresh_cache=True, currency=Currency.USD)
        if code:
            print(f"query account failed：{data}")
            return

        account_data_list = []
        for ix, row in data.iterrows():
            account = FutuAccountData(
                market=self.market,
                env=self.env,
                total_value=float(row['total_assets']),
                cash=float(row['cash'])
            )
            account_data_list.append(account)
        return account_data_list


    def query_position(self):
            """查询持仓"""
            code, data = self.trade_ctx.position_list_query(
                trd_env=self.env, acc_id=0
            )

            if code:
                print(f"query position failed：{data}")
                return None

            position_info_list = []
            for ix, row in data.iterrows():
                position_info = FutuPosition(
                    market=self.market,
                    env=self.env,
                    symbol=symbol_futu_to_snail(row['code']),
                    side=row['position_side'],
                    volume=row['qty'],
                    frozen=float(row['qty'] - float(row['can_sell_qty'])),
                    enter_price=float(row['cost_price']),
                    pnl=row['pl_val']

                )
                position_info_list.append(position_info)

            return position_info_list

    def query_order(self):
        """查询未成交委托"""
        code, data = self.trade_ctx.order_list_query("", trd_env=self.env)

        if code:
            print(f"query order status failed：{data}")
            return

        self.process_order(data)

    def query_trade(self):
        """查询成交"""
        code, data = self.trade_ctx.deal_list_query("", trd_env=self.env)

        if code:
            print(f"query trade status failed：{data}")
            return

        self.process_deal(data)