def copy_batch_data(batch_data, device):
    copy = dict()
    for key in batch_data.keys():
        copy[key] = batch_data[key].clone().to(device)
    return copy


def rearrange_y(meta_data, y, device):
    y_dict = dict()
    for ent_type in meta_data['ent_types']:
        index_pair = meta_data['ent_fault_type_index'][ent_type]
        temp = y[:, meta_data['ent_type_index'][ent_type][0]:meta_data['ent_type_index'][ent_type][1],
               index_pair[0]:index_pair[1]]
        y_dict[ent_type] = temp.reshape(temp.shape[0] * temp.shape[1], temp.shape[2]).contiguous().to(device)
    return y_dict
