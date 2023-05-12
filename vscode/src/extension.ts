import * as vscode from 'vscode';
import axios from 'axios';

export function activate(context: vscode.ExtensionContext) {
	let disposable = vscode.commands.registerCommand('gopilot.autoComplete', async () => {
		const editor = vscode.window.activeTextEditor;
        if (!editor) {
            vscode.window.showErrorMessage('No active editor found.');
            return;
        }

		const document = editor.document;
        const content = document.getText();
        const cursorOffset = document.offsetAt(editor.selection.start);

        try {
            const response = await axios.post('http://localhost:3000/complete', {
                content,
                cursorOffset,
            });

            // Assuming the API returns a list of tokens as plain text
            const tokens = response.data.tokens;

            // Insert the tokens at the cursor position
            const cursorPosition = editor.selection.start;
            await editor.edit((editBuilder) => {
                editBuilder.insert(cursorPosition, tokens);
            });

        } catch (error) {
            vscode.window.showErrorMessage(`Error: ${error}`);
        }

		vscode.window.showInformationMessage('Hello World from Gopilot!');
	});

	context.subscriptions.push(disposable);
}

export function deactivate() {}
