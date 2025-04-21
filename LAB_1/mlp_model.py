from torch import nn


class MLP(nn.Module):
    def __init__(self, input_size, depth, width, classes, residual, normalization):
        super().__init__()

        self.residual = residual

        if normalization:
            self.input_layer = nn.Sequential(
                nn.Linear(input_size, width),
                nn.BatchNorm1d(width),
                nn.ReLU()
            )
        else:
            self.input_layer = nn.Sequential(
                nn.Linear(input_size, width),
                nn.ReLU()
            )

        self.hidden_layers = nn.ModuleList()
        for _ in range(depth):

            if normalization:
                self.hidden_layers.append(
                    nn.Sequential(
                        nn.Linear(width, width),
                        nn.BatchNorm1d(width),
                        nn.ReLU()
                    )
                )
            else:
                self.hidden_layers.append(
                    nn.Sequential(
                        nn.Linear(width, width),
                        nn.ReLU()
                    )
                )

        self.output_layer = nn.Linear(width, classes)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.input_layer(x)

        if self.residual:
            for layer in self.hidden_layers:
                res = x
                x = layer(x)
                x = x + res
        else:
            for layer in self.hidden_layers:
                x = layer(x)

        out = self.output_layer(x)
        return out
