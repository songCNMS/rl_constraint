# constraints = ['G(stock_constraint)',
#                'G(is_replenish_constraint -> ((X!is_replenish_constraint)&(XX!is_replenish_constraint)))',
#                'G(low_profit -> low_stock_constraint)']



def stock_constraint(f_state):
    return (0 < f_state['inventory_in_stock'] <= (f_state['max_vlt']+7)*f_state['sale_mean'])

def is_replenish_constraint(f_state):
    return (f_state['consumption_hist'][-1] > 0)

def low_profit(f_state):
    return ((f_state['sku_price']-f_state['sku_cost']) * f_state['sale_mean'] <= 1000)

def low_stock_constraint(f_state):
    return (0 < f_state['inventory_in_stock'] <= (f_state['max_vlt']+3)*f_state['sale_mean'])

def out_of_stock(f_state):
    return (0 < f_state['inventory_in_stock'])

atoms = {
    'stock_constraint': stock_constraint,
    'is_replenish_constraint': is_replenish_constraint,
    'low_profit': low_profit,
    'low_stock_constraint': low_stock_constraint,
    'out_of_stock': out_of_stock 
}